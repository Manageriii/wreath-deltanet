"""
Stage 2 (v2): TRUE Block Diffusion + NCM + TRM + Multi-Head (H=8) Dual-Gear Wreath LM
=====================================================================================
Honest, end-to-end parallel architecture uniting ALL five pillars on Apple MPS GPU:
  1. SuperBPE + CoBPE Compositional Tokenizer (`E_base + E_mod` input, factorized output).
  2. Work-Efficient GPU Chunk-Parallel Scans (`C = 4`, ZERO token-by-token Python loops!):
     - Gear A: 8x8 {-1, 0, +1} Swap-Boxes (snapped every C=4 tokens in FP8)
     - Gear B: 4x 2D Clock-Dials (snapped to 30° ticks every C=4 tokens in FP8)
     - Multi-Head Worker 2 (`H = 8` parallel query heads reading S_box simultaneously!)
  3. Exp #6 Dependency-Ordered Magnet Cascade:
     - Clock dead-zone & 200-step magnet trigger at >= 70%
     - Downstream Swap-Box magnet held OFF until Clock >= 92% AND Var accuracy >= 82%!
  4. True Block Diffusion + NCM + TRM Output Denoising (Zero 1-by-1 Autoregression!):
     - Entire target response block (`<think> ... </think> <ans> ... </ans>`, B_blk tokens)
       starts as `<MASK>` and denoises IN PARALLEL across 4 coarse-to-fine passes:
       * Pass 0 (NCM Coarse Concept Skeleton): Predicts chunk-level concepts & draft `y_0`
       * Passes 1..3 (Shared Tiny TRM Block): Refines `(z_k, y_k)` across the whole block
         simultaneously—locking `>= 75%` high-confidence `y_k` tokens while keeping latent
         scratchpad `z_k` 100% UNFROZEN!
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp1_adaptive_locking import snap_to_exact_signed_perm
from exp2_deadzone_gate import apply_deadzone_gate
from exp5_fp8_and_clock_gears import quantize_fp8_e4m3
from exp6_trm_ncm_wreath import snap_2d_clock_dial_30deg


# =====================================================================
# GPU Work-Efficient Chunk-Parallel Prefix Scans (ZERO Python t-loops!)
# =====================================================================
def _prefix_matmul_right_to_left(M: torch.Tensor, fp8: bool = False) -> torch.Tensor:
    """Inclusive scan U[..., t] = M[..., t] @ M[..., t-1] @ ... @ M[..., 0] in log2(T) ops."""
    T = M.shape[-3]
    P = quantize_fp8_e4m3(M) if fp8 else M
    st = 1
    while st < T:
        merged = torch.matmul(P[..., st:, :, :], P[..., :-st, :, :])
        if fp8:
            merged = quantize_fp8_e4m3(merged)
        P = torch.cat([P[..., :st, :, :], merged], dim=-3)
        st *= 2
    return P


def gpu_chunk_scan_swap(G: torch.Tensor, chunk_c: int = 4, fp8: bool = False, snap: bool = False) -> torch.Tensor:
    """Chunk-parallel scan for d x d Swap matrices [B, L, d, d] on GPU."""
    B, L, d, _ = G.shape
    pad = (chunk_c - (L % chunk_c)) % chunk_c
    if pad > 0:
        eye_p = torch.eye(d, device=G.device, dtype=G.dtype).view(1, 1, d, d).expand(B, pad, d, d)
        Gp = torch.cat([G, eye_p], dim=1)
    else:
        Gp = G

    N_c = Gp.shape[1] // chunk_c
    Gc = Gp.view(B, N_c, chunk_c, d, d)
    U_intra = _prefix_matmul_right_to_left(Gc, fp8=fp8)

    totals = U_intra[:, :, -1]
    if snap:
        totals = snap_to_exact_signed_perm(totals.reshape(-1, d, d)).view(B, N_c, d, d)

    if N_c > 1:
        U_inter = _prefix_matmul_right_to_left(totals, fp8=False)
        if snap:
            U_inter = snap_to_exact_signed_perm(U_inter.reshape(-1, d, d)).view(B, N_c, d, d)
        eye0 = torch.eye(d, device=G.device, dtype=G.dtype).view(1, 1, d, d).expand(B, 1, d, d)
        prev = torch.cat([eye0, U_inter[:, :-1]], dim=1)
        U_full = torch.matmul(U_intra, prev.unsqueeze(2))
        if fp8:
            U_full = quantize_fp8_e4m3(U_full)
    else:
        U_full = U_intra

    return U_full.reshape(B, -1, d, d)[:, :L]


def gpu_chunk_scan_clock(R: torch.Tensor, chunk_c: int = 4, fp8: bool = False, snap: bool = False) -> torch.Tensor:
    """Chunk-parallel scan for D parallel 2x2 Clock Dials [B, L, D, 2, 2] on GPU."""
    B, L, D, _, _ = R.shape
    Rf = R.permute(0, 2, 1, 3, 4).reshape(B * D, L, 2, 2)
    pad = (chunk_c - (L % chunk_c)) % chunk_c
    if pad > 0:
        eye_p = torch.eye(2, device=R.device, dtype=R.dtype).view(1, 1, 2, 2).expand(B * D, pad, 2, 2)
        Rp = torch.cat([Rf, eye_p], dim=1)
    else:
        Rp = Rf

    N_c = Rp.shape[1] // chunk_c
    Rc = Rp.view(B * D, N_c, chunk_c, 2, 2)
    U_intra = _prefix_matmul_right_to_left(Rc, fp8=fp8)

    totals = U_intra[:, :, -1]
    if snap:
        totals = snap_2d_clock_dial_30deg(totals.reshape(-1, 2, 2)).view(B * D, N_c, 2, 2)
        if fp8:
            totals = quantize_fp8_e4m3(totals)

    if N_c > 1:
        U_inter = _prefix_matmul_right_to_left(totals, fp8=fp8)
        if snap:
            U_inter = snap_2d_clock_dial_30deg(U_inter.reshape(-1, 2, 2)).view(B * D, N_c, 2, 2)
            if fp8:
                U_inter = quantize_fp8_e4m3(U_inter)
        eye0 = torch.eye(2, device=R.device, dtype=R.dtype).view(1, 1, 2, 2).expand(B * D, 1, 2, 2)
        prev = torch.cat([eye0, U_inter[:, :-1]], dim=1)
        U_full = torch.matmul(U_intra, prev.unsqueeze(2))
        if fp8:
            U_full = quantize_fp8_e4m3(U_full)
    else:
        U_full = U_intra

    U_out = U_full.reshape(B, D, -1, 2, 2)[:, :, :L].permute(0, 2, 1, 3, 4)
    return U_out


def gpu_chunk_scan_delta_boxes(
    k_eff: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, chunk_c: int = 16, fp8: bool = False
) -> torch.Tensor:
    """
    Work-efficient GPU chunk-parallel scan for S_t = S_{t-1} A_t + B_t:
      k_eff: [B, L, d_s], v: [B, L, d_v], beta: [B, L, 1] -> returns S_all: [B, L, d_v, d_s]
    """
    B, L, d_s = k_eff.shape
    d_v = v.shape[-1]
    pad = (chunk_c - (L % chunk_c)) % chunk_c
    if pad > 0:
        k_eff = F.pad(k_eff, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))

    Lp = k_eff.shape[1]
    N_c = Lp // chunk_c
    ke = k_eff.view(B, N_c, chunk_c, d_s)
    vc = v.view(B, N_c, chunk_c, d_v)
    bc = beta.view(B, N_c, chunk_c, 1)

    I = torch.eye(d_s, device=k_eff.device, dtype=k_eff.dtype).view(1, 1, 1, d_s, d_s)
    A = I - bc.unsqueeze(-1) * (ke.unsqueeze(-1) * ke.unsqueeze(-2))          # [B, N_c, C, d_s, d_s]
    B_op = (vc * bc).unsqueeze(-1) * ke.unsqueeze(-2)                         # [B, N_c, C, d_v, d_s]
    if fp8:
        A = quantize_fp8_e4m3(A)
        B_op = quantize_fp8_e4m3(B_op)

    st = 1
    while st < chunk_c:
        Al, Bl = A[:, :, :-st], B_op[:, :, :-st]
        Ar, Br = A[:, :, st:], B_op[:, :, st:]
        Am = torch.matmul(Al, Ar)
        Bm = torch.matmul(Bl, Ar) + Br
        if fp8:
            Am, Bm = quantize_fp8_e4m3(Am), quantize_fp8_e4m3(Bm)
        A = torch.cat([A[:, :, :st], Am], dim=2)
        B_op = torch.cat([B_op[:, :, :st], Bm], dim=2)
        st *= 2

    if N_c > 1:
        Ac, Bc = A[:, :, -1], B_op[:, :, -1]  # [B, N_c, ...]
        st = 1
        while st < N_c:
            Al, Bl = Ac[:, :-st], Bc[:, :-st]
            Ar, Br = Ac[:, st:], Bc[:, st:]
            Am = torch.matmul(Al, Ar)
            Bm = torch.matmul(Bl, Ar) + Br
            if fp8:
                Am, Bm = quantize_fp8_e4m3(Am), quantize_fp8_e4m3(Bm)
            Ac = torch.cat([Ac[:, :st], Am], dim=1)
            Bc = torch.cat([Bc[:, :st], Bm], dim=1)
            st *= 2

        z0 = torch.zeros(B, 1, d_v, d_s, device=k_eff.device, dtype=k_eff.dtype)
        S_prev = torch.cat([z0, Bc[:, :-1]], dim=1)  # [B, N_c, d_v, d_s]
        S_full = torch.matmul(S_prev.unsqueeze(2), A) + B_op
        if fp8:
            S_full = quantize_fp8_e4m3(S_full)
    else:
        S_full = B_op

    return S_full.reshape(B, Lp, d_v, d_s)[:, :L]


def gpu_scan_scalar_clock_mem(c_beta: torch.Tensor, w_hand: torch.Tensor, fp8: bool = False) -> torch.Tensor:
    """
    Parallel prefix scan for S_t = (1 - c_beta_t) S_{t-1} + c_beta_t w_hand_t
    c_beta: [B, L, D], w_hand: [B, L, D, 2] -> returns S_clk: [B, L, D, 2]
    """
    B, L, D = c_beta.shape
    a = (1.0 - c_beta).unsqueeze(-1)  # [B, L, D, 1]
    b = c_beta.unsqueeze(-1) * w_hand  # [B, L, D, 2]
    if fp8:
        b = quantize_fp8_e4m3(b)

    st = 1
    while st < L:
        al, bl = a[:, :-st], b[:, :-st]
        ar, br = a[:, st:], b[:, st:]
        am = ar * al
        bm = ar * bl + br
        if fp8:
            bm = quantize_fp8_e4m3(bm)
        a = torch.cat([a[:, :st], am], dim=1)
        b = torch.cat([b[:, :st], bm], dim=1)
        st *= 2
    return b


# =====================================================================
# True Block Diffusion + NCM + TRM + Multi-Head (H = 8) Wreath LM
# =====================================================================
class BlockDiffusionWreathLM(nn.Module):
    """
    1. Encodes arbitrary-length prompt via GPU Chunk-Parallel Wreath Scan with H = 8
       parallel query heads (`use_wreath=True`) vs Standard Multi-Head DeltaNet (`use_wreath=False`).
    2. Generates the entire `<think> ... </think> <ans> ... </ans>` target block (`B_blk` tokens)
       IN PARALLEL via Coarse-to-Fine NCM + TRM Block Diffusion:
         - Pass 0 (NCM): Coarse chunk concepts predict initial block draft `y_0` from `<MASK>`
         - Passes 1..3 (TRM): Shared recursive block refines `(z_k, y_k)` bidirectionally
           inside the active block while cross-reading the `H = 8` Wreath state!
           (`y_k` locks `>= 75%` confident tokens; `z_k` stays 100% unfrozen!)
    """

    def __init__(
        self,
        base_vocab: int,
        mod_vocab: int,
        mask_base_id: int,
        d_model: int = 128,
        n_heads: int = 8,
        d_swap: int = 8,
        d_v: int = 32,
        n_dials: int = 4,
        chunk_c: int = 4,
        block_len: int = 24,
        use_wreath_trm: bool = True,
    ):
        super().__init__()
        self.base_vocab = base_vocab
        self.mod_vocab = mod_vocab
        self.mask_base_id = mask_base_id
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_swap = d_swap
        self.d_v = d_v
        self.n_dials = n_dials
        self.chunk_c = chunk_c
        self.block_len = block_len
        self.use_wreath_trm = use_wreath_trm

        # CoBPE Compositional Embeddings
        self.base_emb = nn.Embedding(base_vocab, d_model)
        self.mod_emb = nn.Embedding(mod_vocab, d_model)

        # Causal 4-token Depthwise Conv1d for prompt phrase binding (`set a=` + ` 4`)
        self.prompt_conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.prompt_conv_proj = nn.Linear(d_model, d_model)

        # ---- PROMPT ENCODER: Multi-Head (H = 8) Worker 2 + Dual-Gear Worker 1 ----
        self.w2_norm = nn.LayerNorm(d_model)
        self.to_q_heads = nn.Linear(d_model, n_heads * d_swap)  # H = 8 PARALLEL QUERY HEADS!
        self.to_k = nn.Linear(d_model, d_swap)
        self.to_v = nn.Linear(d_model, d_v)
        self.to_beta = nn.Linear(d_model, 1)
        with torch.no_grad():
            # Standard SSM/RNN near-Identity init so end-of-prompt gradients don't vanish as 0.75^56!
            self.to_beta.bias.fill_(-2.2)

        if use_wreath_trm:
            self.w1_norm = nn.LayerNorm(d_model)
            self.to_hh_k = nn.Linear(d_model, 2 * d_swap)
            self.to_hh_b = nn.Linear(d_model, 2)
            self.to_clk_theta = nn.Linear(d_model, n_dials)
            self.to_clk_phi = nn.Linear(d_model, n_dials)
            self.to_clk_beta = nn.Linear(d_model, n_dials)
            with torch.no_grad():
                self.to_hh_b.bias.fill_(-2.5)
                self.to_clk_beta.bias.fill_(-2.2)
                self.to_clk_theta.weight.mul_(0.08)
                self.to_clk_theta.bias.zero_()
            prompt_summary_dim = n_heads * d_v + 2 * n_dials + d_model
        else:
            prompt_summary_dim = n_heads * d_v + d_model

        self.prompt_state_proj = nn.Sequential(
            nn.LayerNorm(prompt_summary_dim),
            nn.Linear(prompt_summary_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        # Project fixed O(1) Memory Slots (H=8 Box-Head slots + D=4 Clock slots + 1 anchor) into d_model
        # -> ZERO O(L_prompt) KV cache! 100% sub-quadratic O(1) memory interface!
        self.box_slot_proj = nn.Linear(d_v, d_model)
        self.slot_type_emb = nn.Embedding(16, d_model)
        self.mem_kv = nn.Linear(d_model, 2 * d_model)

        if use_wreath_trm:
            self.clk_slot_proj = nn.Linear(2, d_model)
            ctx_in_dim = d_model + 2 * n_dials + d_v
            self.w1_cascade_fusion = nn.Sequential(
                nn.LayerNorm(ctx_in_dim),
                nn.Linear(ctx_in_dim, d_model),
                nn.SiLU(),
            )

        # ---- BLOCK DIFFUSION + NCM + TRM DENOISER ----
        self.block_pos_emb = nn.Parameter(torch.randn(1, block_len, d_model) * 0.02)
        self.pass_emb = nn.Embedding(4, d_model)

        # Stage 1 (NCM): Coarse Concept Planner predicts `block_len // 4` concept vectors from prompt state!
        n_concepts = block_len // chunk_c
        self.ncm_concept_planner = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_concepts * d_model),
        )

        # Shared Tiny TRM Denoising Block (bidirectional inside the `block_len` block + cross-reads prompt!)
        self.trm_norm1 = nn.LayerNorm(2 * d_model)
        self.trm_in_proj = nn.Linear(2 * d_model, d_model)
        self.block_self_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.prompt_cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.trm_z_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.base_head = nn.Linear(d_model, base_vocab)
        self.mod_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, mod_vocab),
        )

    def encode_prompt_parallel(
        self,
        p_base: torch.Tensor,
        p_mod: torch.Tensor,
        box_noise: float = 0.0,
        clk_noise: float = 0.0,
        box_dz: float = 0.0,
        clk_dz: float = 0.0,
        fp8_eval: bool = False,
        snap_swap: bool = False,
        snap_clk: bool = False,
    ):
        B, Lp = p_base.shape
        not_pad = (p_base != 0).float().unsqueeze(-1)  # [B, Lp, 1] -> <PAD> is an exact mathematical no-op!
        raw = (self.base_emb(p_base) + self.mod_emb(p_mod)) * not_pad
        conv = F.silu(self.prompt_conv(raw.transpose(1, 2))[:, :, :Lp].transpose(1, 2)) * not_pad
        h = raw + self.prompt_conv_proj(conv) * not_pad

        w2 = self.w2_norm(h)
        q_heads = F.normalize(self.to_q_heads(w2).view(B, Lp, self.n_heads, self.d_swap), dim=-1)
        k = F.normalize(self.to_k(w2), dim=-1)
        if box_dz > 0.0:
            k_trim = torch.where(k.abs() < 0.20, torch.zeros_like(k), k)
            k = k_trim / k_trim.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        v = self.to_v(w2)
        b_beta = apply_deadzone_gate(self.to_beta(w2), zone=box_dz, active=(box_dz > 0.0)) * not_pad

        if self.use_wreath_trm:
            # ---- PROMPT STAGE A: Run 2D Clock-Dial + Un-swapped Skeleton Box Scan FIRST on GPU! ----
            w1_base = self.w1_norm(h + 0.5 * self.mod_emb(p_mod) * not_pad)
            rth = self.to_clk_theta(w1_base)
            if self.training and clk_noise > 0.0:
                rth = rth + torch.randn_like(rth) * (clk_noise * 0.5)

            theta = (math.pi * torch.tanh(rth)) * not_pad
            phi = math.pi * torch.tanh(self.to_clk_phi(w2))
            c_beta = apply_deadzone_gate(self.to_clk_beta(w2), zone=clk_dz, active=(clk_dz > 0.0)) * not_pad
            clk_mag = (0.5 * (1.0 - torch.cos(12.0 * theta))).mean() + (
                c_beta.detach() * 0.5 * (1.0 - torch.cos(12.0 * phi))
            ).mean()

            ct, st = torch.cos(theta), torch.sin(theta)
            R_clk = torch.stack([torch.stack([ct, -st], dim=-1), torch.stack([st, ct], dim=-1)], dim=-2)
            U_clk = gpu_chunk_scan_clock(R_clk, chunk_c=self.chunk_c, fp8=fp8_eval, snap=snap_clk)

            reset_hand = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
            w_hand = torch.matmul(U_clk.transpose(-1, -2), reset_hand.unsqueeze(-1)).squeeze(-1)
            S_clk_all = gpu_scan_scalar_clock_mem(c_beta, w_hand, fp8=fp8_eval)
            # Current clock hand at every prompt token: [B, Lp, 2*D]
            clk_seq = torch.matmul(U_clk, S_clk_all.unsqueeze(-1)).squeeze(-1).reshape(B, Lp, -1)

            # Skeleton un-swapped box scan so `"if Ava even"` also sees `Ava`!
            S_skel = gpu_chunk_scan_delta_boxes(k, v, b_beta, chunk_c=16, fp8=fp8_eval)
            q0 = q_heads[:, :, 0]  # Head 0 probes control variable (`Ava` / `Box 0`)!
            box_skel_seq = torch.matmul(S_skel, q0.unsqueeze(-1)).squeeze(-1)  # [B, Lp, d_v]

            # Shift both by 1 token causally so token `t` sees state at `t - 1`!
            zpad_clk = torch.zeros(B, 1, 2 * self.n_dials, device=h.device)
            zpad_box = torch.zeros(B, 1, self.d_v, device=h.device)
            prev_state_ctx = torch.cat(
                [w1_base, torch.cat([zpad_clk, clk_seq[:, :-1]], dim=1), torch.cat([zpad_box, box_skel_seq[:, :-1]], dim=1)],
                dim=-1,
            )

            # ---- PROMPT STAGE B: Conditional Wreath Swapper (now sees `clk` & `Ava`!) ----
            w1 = self.w1_cascade_fusion(prev_state_ctx) * not_pad
            rk = self.to_hh_k(w1).view(B, Lp, 2, self.d_swap)
            rb = self.to_hh_b(w1)
            if self.training and box_noise > 0.0:
                rk = rk + torch.randn_like(rk) * box_noise
                rb = rb + torch.randn_like(rb) * box_noise

            hk = F.normalize(rk, dim=-1)
            hb = (2.0 * torch.sigmoid(rb)) * not_pad
            Id = torch.eye(self.d_swap, device=h.device).view(1, 1, self.d_swap, self.d_swap)
            H1 = Id - hb[:, :, 0:1].unsqueeze(-1) * (hk[:, :, 0].unsqueeze(-1) * hk[:, :, 0].unsqueeze(-2))
            H2 = Id - hb[:, :, 1:2].unsqueeze(-1) * (hk[:, :, 1].unsqueeze(-1) * hk[:, :, 1].unsqueeze(-2))
            G_swap = torch.matmul(H2, H1)

            ortho_v = (((G_swap ** 2).sum(dim=-1) - 1.0) ** 2).mean()
            pm1_v = (self.d_swap - (G_swap ** 4).sum(dim=(-2, -1))).mean() / self.d_swap
            addr_v = (b_beta.detach().squeeze(-1) * (1.0 - (k ** 4).sum(dim=-1))).mean()
            box_mag = ortho_v + pm1_v + addr_v

            self.last_G_swap = G_swap.detach()
            self.last_theta = theta.detach()

            U_swap = gpu_chunk_scan_swap(G_swap, chunk_c=self.chunk_c, fp8=fp8_eval, snap=snap_swap)
            k_eff = torch.matmul(U_swap.transpose(-1, -2), k.unsqueeze(-1)).squeeze(-1)
            S_all = gpu_chunk_scan_delta_boxes(k_eff, v, b_beta, chunk_c=16, fp8=fp8_eval)

            # Read ALL H = 8 Query Heads at the end of the prompt simultaneously!
            U_last_T = U_swap[:, -1].transpose(-1, -2)
            q_last_T = q_heads[:, -1].transpose(-1, -2)
            q_eff_T = torch.bmm(U_last_T, q_last_T)
            box_heads_mat = torch.bmm(S_all[:, -1], q_eff_T).transpose(1, 2)  # [B, H=8, d_v]
            read_8heads = box_heads_mat.reshape(B, -1)
            clk_final_mat = clk_seq[:, -1].view(B, self.n_dials, 2)           # [B, D=4, 2]
            clk_final = clk_seq[:, -1]

            # Build fixed O(1) memory slot bank: [8 Box Slots, 4 Clock Slots, 1 Last-Token Slot] = 13 slots!
            b_slots = self.box_slot_proj(box_heads_mat)
            c_slots = self.clk_slot_proj(clk_final_mat)
            mem_slots = torch.cat([b_slots, c_slots, h[:, -1:]], dim=1)       # [B, 13, d_model]
            slot_ids = torch.arange(mem_slots.shape[1], device=h.device).unsqueeze(0)
            mem_slots = mem_slots + self.slot_type_emb(slot_ids)

            summary = torch.cat([read_8heads, clk_final, h[:, -1]], dim=-1)
        else:
            box_mag = torch.tensor(0.0, device=h.device)
            clk_mag = torch.tensor(0.0, device=h.device)
            S_all = gpu_chunk_scan_delta_boxes(k, v, b_beta, chunk_c=16, fp8=fp8_eval)
            q_last_T = q_heads[:, -1].transpose(-1, -2)
            box_heads_mat = torch.bmm(S_all[:, -1], q_last_T).transpose(1, 2)  # [B, H=8, d_v]
            read_8heads = box_heads_mat.reshape(B, -1)

            b_slots = self.box_slot_proj(box_heads_mat)
            mem_slots = torch.cat([b_slots, h[:, -1:]], dim=1)                 # [B, 9, d_model]
            slot_ids = torch.arange(mem_slots.shape[1], device=h.device).unsqueeze(0)
            mem_slots = mem_slots + self.slot_type_emb(slot_ids)

            summary = torch.cat([read_8heads, h[:, -1]], dim=-1)

        p_state = self.prompt_state_proj(summary)  # [B, d_model]
        return mem_slots, p_state, box_mag, clk_mag

    def denoise_step(
        self,
        y_emb: torch.Tensor,
        z: torch.Tensor,
        mem_slots: torch.Tensor,
        pass_idx: int,
    ):
        x = self.trm_in_proj(self.trm_norm1(torch.cat([y_emb, z], dim=-1))) + self.block_pos_emb + self.pass_emb.weight[pass_idx].view(1, 1, -1)
        # Bidirectional self-attention strictly inside the small 32-token denoising block
        attn_self, _ = self.block_self_attn(x, x, x, need_weights=False)
        x = x + attn_self
        # Cross-read ONLY the fixed O(1) recurrent memory slots (13 vectors -> ZERO O(L_prompt) cache!)
        mk, mv = self.mem_kv(mem_slots).chunk(2, dim=-1)
        attn_cross, _ = self.prompt_cross_attn(x, mk, mv, need_weights=False)
        x = x + attn_cross

        # Unfrozen TRM Scratchpad update!
        z_next = z + self.trm_z_mlp(x)
        h_out = self.out_norm(z_next)
        b_logits = self.base_head(h_out)
        # Always condition modifier on model's own predicted base distribution -> zero train/eval mismatch!
        pred_base_e = torch.matmul(F.softmax(b_logits, dim=-1), self.base_emb.weight)
        m_logits = self.mod_head(torch.cat([h_out, pred_base_e], dim=-1))
        return z_next, b_logits, m_logits

    def forward(
        self,
        p_base: torch.Tensor,
        p_mod: torch.Tensor,
        tgt_base: torch.Tensor | None = None,
        box_noise: float = 0.0,
        clk_noise: float = 0.0,
        box_dz: float = 0.0,
        clk_dz: float = 0.0,
        fp8_eval: bool = False,
        snap_swap: bool = False,
        snap_clk: bool = False,
    ):
        B = p_base.shape[0]
        mem_slots, p_state, b_mag, c_mag = self.encode_prompt_parallel(
            p_base,
            p_mod,
            box_noise=box_noise,
            clk_noise=clk_noise,
            box_dz=box_dz,
            clk_dz=clk_dz,
            fp8_eval=fp8_eval,
            snap_swap=snap_swap,
            snap_clk=snap_clk,
        )

        # ---- PASS 0 (NCM Coarse Concept Skeleton): Expand coarse chunk concepts -> initial z_0, y_0 ----
        n_c = self.block_len // self.chunk_c
        coarse_concepts = self.ncm_concept_planner(p_state).view(B, n_c, 1, self.d_model)
        z = coarse_concepts.expand(B, n_c, self.chunk_c, self.d_model).reshape(B, self.block_len, self.d_model)
        mask_emb = self.base_emb.weight[self.mask_base_id].view(1, 1, -1).expand(B, self.block_len, -1)

        z, b_log0, m_log0 = self.denoise_step(mask_emb, z, mem_slots, pass_idx=0)
        b_logs = [b_log0]
        m_logs = [m_log0]

        n_trm_passes = 3 if self.use_wreath_trm else 1
        for k_pass in range(1, n_trm_passes + 1):
            probs_b = F.softmax(b_logs[-1], dim=-1)
            probs_m = F.softmax(m_logs[-1], dim=-1)
            conf_b, pred_b = probs_b.max(dim=-1, keepdim=True)
            pred_m = probs_m.argmax(dim=-1)

            soft_e = torch.matmul(probs_b, self.base_emb.weight) + torch.matmul(probs_m, self.mod_emb.weight)
            hard_e = self.base_emb(pred_b.squeeze(-1)) + self.mod_emb(pred_m)
            draft_e = hard_e - soft_e.detach() + soft_e

            target_w = torch.where(conf_b >= 0.75, torch.ones_like(conf_b), conf_b)
            w = target_w - conf_b.detach() + conf_b
            y_locked = w * draft_e + (1.0 - w) * mask_emb

            z, bl, ml = self.denoise_step(y_locked, z, mem_slots, pass_idx=k_pass)
            b_logs.append(bl)
            m_logs.append(ml)

        return b_logs, m_logs, b_mag, c_mag
