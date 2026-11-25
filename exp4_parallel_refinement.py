"""
Experiment #4: Parallel Tournament-Bracket Chunk Engine + 3-Pass Self-Correcting Refinement
===========================================================================================
Part A: Benchmarks true Parallel Chunk + Tournament-Bracket Scan vs. Sequential Loop
        at L = 8,192 in BF16 across chunk sizes C = 8, 64, 512, 8192.
Part B: Tests our 3-Pass Image-Generator-Style Parallel Refinement using ONE compact
        weight-shared block on a 2-Hop Chained Conditional Swap Puzzle:
          - Rule 1: IF Box 0 is EVEN -> Swap Box 1 <-> Box 2
          - Rule 2: IF Box 1 is EVEN -> Swap Box 3 <-> Box 4  (depends on Box 1 AFTER swaps!)
        Across 3 parallel passes:
          Pass 1 (Skeleton) -> Pass 2 (Full Draft + Confidence Score) -> Pass 3 (Remask & Refine)
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_adaptive_locking import (
    compute_adaptive_control,
    generate_noisy_text_puzzle,
    snap_to_exact_signed_perm,
)
from exp2_deadzone_gate import WreathDeltaNetV2, apply_deadzone_gate


# =====================================================================
# Parallel Tournament-Bracket Scan Helpers (Log2 Rounds Instead of L Steps!)
# =====================================================================
def parallel_prefix_matmul(M: torch.Tensor) -> torch.Tensor:
    """
    Inclusive prefix matrix product U[..., t] = M[..., t] @ M[..., t-1] @ ... @ M[..., 0]
    computed in ceil(log2(T)) parallel doubling rounds (Hillis-Steele tournament)!
    M shape: [..., T, d, d]
    """
    T = M.shape[-3]
    P = M
    stride = 1
    while stride < T:
        left = P[..., :-stride, :, :]
        right = P[..., stride:, :, :]
        merged = torch.matmul(right, left)
        P = torch.cat([P[..., :stride, :, :], merged], dim=-3)
        stride *= 2
    return P


def parallel_chunked_group_scan(G: torch.Tensor, chunk_size: int, dtype=torch.bfloat16, snap: bool = True) -> torch.Tensor:
    """
    Splits L into (L // chunk_size) chunks, runs all chunks in parallel, snaps chunk
    summaries to exact {-1, 0, +1}, and combines chunk summaries in log2 rounds!
    G: [B, L, d, d] -> returns U: [B, L, d, d]
    """
    B, L, d, _ = G.shape
    if L % chunk_size != 0:
        pad = chunk_size - (L % chunk_size)
        eye_pad = torch.eye(d, device=G.device, dtype=G.dtype).view(1, 1, d, d).expand(B, pad, d, d)
        G_padded = torch.cat([G, eye_pad], dim=1)
    else:
        pad = 0
        G_padded = G

    N_chunks = G_padded.shape[1] // chunk_size
    G_chunks = G_padded.to(dtype).view(B, N_chunks, chunk_size, d, d)

    # Stage 1: All N_chunks run intra-chunk tournament scan simultaneously!
    U_intra = parallel_prefix_matmul(G_chunks)  # [B, N_chunks, C, d, d]

    # Snap each chunk's final summary to exact {-1, 0, +1}
    chunk_totals = U_intra[:, :, -1]  # [B, N_chunks, d, d]
    if snap:
        chunk_totals = snap_to_exact_signed_perm(chunk_totals.float().reshape(-1, d, d)).view(B, N_chunks, d, d).to(dtype)

    # Stage 2: Combine all N_chunks summaries in log2(N_chunks) tournament rounds!
    if N_chunks > 1:
        # Since snapped summaries are exact {-1, 0, +1}, compute in float32/int-exact so 0 drift!
        U_inter = parallel_prefix_matmul(chunk_totals.float())  # [B, N_chunks, d, d]
        if snap:
            U_inter = snap_to_exact_signed_perm(U_inter.reshape(-1, d, d)).view(B, N_chunks, d, d)
        eye = torch.eye(d, device=G.device, dtype=torch.float32).view(1, 1, d, d).expand(B, 1, d, d)
        prev_totals = torch.cat([eye, U_inter[:, :-1]], dim=1).to(dtype)  # [B, N_chunks, d, d]
        # Broadcast chunk start state into every position inside each chunk in 1 parallel op!
        U_full = torch.matmul(U_intra, prev_totals.unsqueeze(2))
    else:
        U_full = U_intra

    U_out = U_full.reshape(B, -1, d, d)
    if pad > 0:
        U_out = U_out[:, :L]
    return U_out.float()


def parallel_affine_delta_scan(k_eff: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, dtype=torch.bfloat16):
    """
    Parallel tournament prefix scan for Worker 2 (Delta-Rule Value Boxes):
      S_t = S_{t-1} A_t + B_t
    Associative merge rule:
      (A_left, B_left) (+) (A_right, B_right) = (A_left @ A_right, B_left @ A_right + B_right)
    Runs in ceil(log2(L)) vectorized rounds instead of L sequential steps!
    """
    B_sz, L, d_g = k_eff.shape
    d_v = v.shape[-1]
    kc = k_eff.to(dtype)
    vc = v.to(dtype)
    bc = beta.to(dtype)  # [B, L, 1]

    I = torch.eye(d_g, device=k_eff.device, dtype=dtype).view(1, 1, d_g, d_g)
    # A_t = I - beta_t k_t k_t^T  [B, L, d_g, d_g]
    A = I - bc.unsqueeze(-1) * torch.einsum("bti,btj->btij", kc, kc)
    # B_t = beta_t v_t k_t^T      [B, L, d_v, d_g]
    B_op = torch.einsum("btv,btg->btvg", vc * bc, kc)

    stride = 1
    while stride < L:
        A_left = A[:, :-stride]
        B_left = B_op[:, :-stride]
        A_right = A[:, stride:]
        B_right = B_op[:, stride:]

        A_merged = torch.matmul(A_left, A_right)
        B_merged = torch.matmul(B_left, A_right) + B_right

        A = torch.cat([A[:, :stride], A_merged], dim=1)
        B_op = torch.cat([B_op[:, :stride], B_merged], dim=1)
        stride *= 2

    return B_op.float()  # Since S_0 = 0, S_t = 0 @ A_{1:t} + B_{1:t} = B_op[:, t]!


# =====================================================================
# PART A: Speed & Goldilocks Chunk Size Benchmark at L = 8,192 (BF16)
# =====================================================================
def run_part_a_chunk_benchmark():
    print("=" * 86)
    print("PART 4A: Parallel Chunk Tournament Engine vs. Sequential Loop (L = 8,192 in BF16)")
    print("=" * 86)

    torch.manual_seed(12)
    model = WreathDeltaNetV2(vocab_size=64, d_g=6, d_v=24, num_vals=8)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
    recent_acc = 0.125

    # Quick train (same setup as Exp #2) so we benchmark a real trained model!
    for step in range(1, 901):
        lam, noise_std = compute_adaptive_control(step, recent_acc)
        dz = 0.05 if recent_acc >= 0.75 else 0.0
        toks, tgts = generate_noisy_text_puzzle(64, seq_len=48)
        logits, m_loss = model(toks, noise_std=noise_std, deadzone=dz)
        loss = F.cross_entropy(logits[:, 5:].reshape(-1, 8), tgts[:, 5:].reshape(-1)) + lam * m_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            b_acc = (logits[:, -1].argmax(dim=-1) == tgts[:, -1]).float().mean().item()
            recent_acc = 0.90 * recent_acc + 0.10 * b_acc

    model.eval()
    torch.manual_seed(999)
    L_bench = 8192
    etoks, etgts = generate_noisy_text_puzzle(32, L_bench)

    with torch.no_grad():
        # 1. Sequential single-file loop baseline
        t0 = time.perf_counter()
        seq_logits, _ = model(etoks, deadzone=0.05, dtype=torch.bfloat16, chunk_snap=True, final_only=True)
        seq_ms = (time.perf_counter() - t0) * 1000.0
        seq_acc = (seq_logits[:, 0].argmax(dim=-1) == etgts[:, -1]).float().mean().item()
        print(f"  Sequential Single-File Loop (8,192 steps) -> Time: {seq_ms:7.1f} ms | BF16 Acc: {seq_acc*100:5.1f}%")

        # Precompute token projections once for fair chunk-size sweep
        for C in [8, 64, 512, 8192]:
            t0 = time.perf_counter()
            x = model.emb(etoks)
            hk = F.normalize(model.to_hh_k(x).view(32, L_bench, 2, 6), dim=-1)
            hb = 2.0 * torch.sigmoid(model.to_hh_b(x))
            I = torch.eye(6).view(1, 1, 6, 6)
            H1 = I - hb[:, :, 0:1].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 0], hk[:, :, 0])
            H2 = I - hb[:, :, 1:2].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 1], hk[:, :, 1])
            G = torch.matmul(H2, H1)

            # Note: When C=8192, NO snap happens until step 8192 (drift piles up inside the single giant chunk!)
            U = parallel_chunked_group_scan(G, chunk_size=C, dtype=torch.bfloat16, snap=(C < 8192))
            k = F.normalize(model.to_k(x), dim=-1)
            v = model.to_v(x)
            beta = apply_deadzone_gate(model.to_beta(x), zone=0.05, active=True)
            k_eff = torch.einsum("btij,bti->btj", U, k)
            S_all = parallel_affine_delta_scan(k_eff, v, beta, dtype=torch.bfloat16)

            q_norm = F.normalize(model.box_queries, dim=-1)
            q_final = torch.einsum("bij,ri->brj", U[:, -1], q_norm)
            p_logits = model.readout(torch.einsum("bvg,brg->brv", S_all[:, -1], q_final))
            par_ms = (time.perf_counter() - t0) * 1000.0
            par_acc = (p_logits.argmax(dim=-1) == etgts[:, -1]).float().mean().item()
            speedup = seq_ms / par_ms
            n_chunks = L_bench // C
            print(
                f"  Parallel Chunk Engine (C={C:4d}, {n_chunks:4d} chunks) -> Time: {par_ms:7.1f} ms ({speedup:4.1f}x faster!) | BF16 Acc: {par_acc*100:5.1f}%"
            )


# =====================================================================
# PART B: 3-Pass Self-Correcting Refinement (1 Compact Shared Block!)
# =====================================================================
def generate_two_hop_chained_puzzle(batch_size: int, seq_len: int, num_vals: int = 8):
    """
    2-Hop Chained Conditional Puzzle (Needs 3-level reasoning depth!):
      - Token 0, 1 : Unconditional swaps (1<->2), (3<->4)
      - Token 2    : HOP-1 CONDITIONAL: IF Box 0 is EVEN -> Swap Box 1 <-> Box 2
      - Token 3    : HOP-2 CHAINED CONDITIONAL: IF Box 1 is EVEN -> Swap Box 3 <-> Box 4!
                     (Notice: knowing Box 1 requires FIRST resolving Rule 1's conditional swap!)
      - Token 4..23: 20 filler chatter words
      - Token 24..63: Write(box r, val v) for r in 0..4
    """
    write_offset = 24
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    init_vals = torch.randint(0, num_vals, (batch_size, 5))
    for r in range(5):
        tokens[:, r] = write_offset + r * num_vals + init_vals[:, r]

    u = torch.rand(batch_size, seq_len - 5)
    fillers = 4 + torch.randint(0, 20, (batch_size, seq_len - 5))
    uncond = torch.randint(0, 2, (batch_size, seq_len - 5))
    hop1 = torch.full((batch_size, seq_len - 5), 2, dtype=torch.long)
    hop2 = torch.full((batch_size, seq_len - 5), 3, dtype=torch.long)
    writes = write_offset + torch.randint(0, 5 * num_vals, (batch_size, seq_len - 5))

    rest = torch.where(
        u < 0.70,
        fillers,
        torch.where(u < 0.78, uncond, torch.where(u < 0.86, hop1, torch.where(u < 0.94, hop2, writes))),
    )
    tokens[:, 5:] = rest

    boxes = torch.zeros(batch_size, 5, dtype=torch.long)
    targets = torch.zeros(batch_size, seq_len, 5, dtype=torch.long)
    for t in range(seq_len):
        tk = tokens[:, t]
        m0 = tk == 0
        if m0.any():
            boxes[m0, 1], boxes[m0, 2] = boxes[m0, 2].clone(), boxes[m0, 1].clone()
        m1 = tk == 1
        if m1.any():
            boxes[m1, 3], boxes[m1, 4] = boxes[m1, 4].clone(), boxes[m1, 3].clone()
        c1 = (tk == 2) & ((boxes[:, 0] % 2) == 0)
        if c1.any():
            boxes[c1, 1], boxes[c1, 2] = boxes[c1, 2].clone(), boxes[c1, 1].clone()
        c2 = (tk == 3) & ((boxes[:, 1] % 2) == 0)
        if c2.any():
            boxes[c2, 3], boxes[c2, 4] = boxes[c2, 4].clone(), boxes[c2, 3].clone()
        wm = tk >= write_offset
        if wm.any():
            wt = tk[wm] - write_offset
            boxes[wm, wt // num_vals] = wt % num_vals
        targets[:, t] = boxes
    return tokens, targets


class ThreePassRefinementWreath(nn.Module):
    """
    ONE SINGLE COMPACT BLOCK reused across 3 Parallel Refinement Passes:
      - Pass 1 (Skeleton)        : Swapper OFF -> clean un-swapped draft of Box 0 & raw writes
      - Pass 2 (Full Draft)      : Swapper ON  -> uses Skeleton, predicts all boxes + Confidence Score!
      - Pass 3 (Remask & Refine) : Remasks low-confidence draft slots (< conf_thresh) back to Pass 1's
                                   clean skeleton while locking high-confidence Box 1 anchors, solving Hop 2!
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 64, d_g: int = 6, d_v: int = 24, num_vals: int = 8):
        super().__init__()
        self.d_g = d_g
        self.d_v = d_v
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Learnable embedding encoding the model's previous-pass box draft back into the next pass!
        self.draft_proj = nn.Linear(5 * num_vals, d_model)
        self.pass_emb = nn.Embedding(3, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.to_hh_k = nn.Linear(d_model, 2 * d_g)
        self.to_hh_b = nn.Linear(d_model, 2)

        self.to_k = nn.Linear(d_model, d_g)
        self.to_v = nn.Linear(d_model, d_v)
        self.to_beta = nn.Linear(d_model, 1)
        self.box_queries = nn.Parameter(torch.eye(d_g)[:5])

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_v),
            nn.Linear(d_v, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_vals),
        )

    def single_parallel_pass(
        self,
        h: torch.Tensor,
        use_swapper: bool,
        noise_std: float = 0.0,
        deadzone: float = 0.0,
        dtype=torch.float32,
        chunk_snap: bool = False,
    ):
        B, L, _ = h.shape
        x = self.norm1(h)

        if use_swapper:
            raw_k = self.to_hh_k(x).view(B, L, 2, self.d_g)
            raw_b = self.to_hh_b(x)
            if self.training and noise_std > 0.0:
                raw_k = raw_k + torch.randn_like(raw_k) * noise_std
                raw_b = raw_b + torch.randn_like(raw_b) * noise_std
            hk = F.normalize(raw_k, dim=-1)
            hb = 2.0 * torch.sigmoid(raw_b)
            I = torch.eye(self.d_g, device=x.device).view(1, 1, self.d_g, self.d_g)
            H1 = I - hb[:, :, 0:1].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 0], hk[:, :, 0])
            H2 = I - hb[:, :, 1:2].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 1], hk[:, :, 1])
            G = torch.matmul(H2, H1)

            ortho_valley = (((G ** 2).sum(dim=-1) - 1.0) ** 2).mean()
            pm1_valley = (self.d_g - (G ** 4).sum(dim=(-2, -1))).mean() / self.d_g
            magnet_loss = ortho_valley + pm1_valley

            U = parallel_chunked_group_scan(G, chunk_size=64, dtype=dtype, snap=chunk_snap)
        else:
            magnet_loss = torch.tensor(0.0, device=h.device)
            U = torch.eye(self.d_g, device=h.device).view(1, 1, self.d_g, self.d_g).expand(B, L, self.d_g, self.d_g)

        k = F.normalize(self.to_k(x), dim=-1)
        v = self.to_v(x)
        beta = apply_deadzone_gate(self.to_beta(x), zone=deadzone, active=(deadzone > 0.0))

        k_eff = torch.einsum("btij,bti->btj", U, k)
        S_all = parallel_affine_delta_scan(k_eff, v, beta, dtype=dtype)  # [B, L, d_v, d_g]

        q_norm = F.normalize(self.box_queries, dim=-1)
        q_eff = torch.einsum("btij,ri->btrj", U, q_norm)  # [B, L, 5, d_g]
        box_feats = torch.einsum("btvg,btrg->btrv", S_all, q_eff)  # [B, L, 5, d_v]
        logits = self.mlp(box_feats)  # [B, L, 5, num_vals]
        return logits, magnet_loss

    def forward(
        self,
        tokens: torch.Tensor,
        noise_std: float = 0.0,
        deadzone: float = 0.0,
        dtype=torch.float32,
        chunk_snap: bool = False,
        conf_threshold: float = 0.75,
    ):
        B, L = tokens.shape
        base_h = self.tok_emb(tokens)

        # ---- PASS 1: SKELETON (Swapper OFF -> clean draft of un-swapped control states) ----
        h1 = base_h + self.pass_emb.weight[0].view(1, 1, -1)
        logits_p1, _ = self.single_parallel_pass(
            h1, use_swapper=False, noise_std=0.0, deadzone=deadzone, dtype=dtype, chunk_snap=False
        )
        probs_p1 = F.softmax(logits_p1, dim=-1)  # [B, L, 5, 8]
        # Note: causal shift by 1 token so step `t` sees the state of boxes JUST BEFORE token `t` acts!
        zero_pad = torch.zeros(B, 1, 5, 8, device=tokens.device)
        skeleton_ctx = torch.cat([zero_pad, probs_p1[:, :-1]], dim=1)

        # ---- PASS 2: FULL DRAFT + SCORING FORMULA (Swapper ON) ----
        h2 = base_h + self.draft_proj(skeleton_ctx.reshape(B, L, -1)) + self.pass_emb.weight[1].view(1, 1, -1)
        logits_p2, m_loss2 = self.single_parallel_pass(
            h2, use_swapper=True, noise_std=noise_std, deadzone=deadzone, dtype=dtype, chunk_snap=chunk_snap
        )
        probs_p2 = F.softmax(logits_p2, dim=-1)
        draft_ctx = torch.cat([zero_pad, probs_p2[:, :-1]], dim=1)  # [B, L, 5, 8]

        # SCORING FORMULA: Confidence = max class probability of each box in Pass 2's draft!
        confidence = draft_ctx.max(dim=-1, keepdim=True).values  # [B, L, 5, 1] in [0.125, 1.0]

        # ---- PASS 3: SELF-CORRECTING REMASKING ----
        # High-confidence box drafts (>= conf_threshold) stay locked in;
        # Low-confidence uncertain box drafts (< conf_threshold) are REMASKED back to Pass 1's clean skeleton!
        keep_mask = (confidence >= conf_threshold).float()
        remasked_ctx = keep_mask * draft_ctx + (1.0 - keep_mask) * skeleton_ctx.detach()

        h3 = base_h + self.draft_proj(remasked_ctx.reshape(B, L, -1)) + self.pass_emb.weight[2].view(1, 1, -1)
        logits_p3, m_loss3 = self.single_parallel_pass(
            h3, use_swapper=True, noise_std=noise_std, deadzone=deadzone, dtype=dtype, chunk_snap=chunk_snap
        )

        return [logits_p1, logits_p2, logits_p3], 0.5 * (m_loss2 + m_loss3)


def run_part_b_three_pass_refinement():
    print("\n" + "=" * 86)
    print("PART 4B: 3-Pass Parallel Refinement (Skeleton -> Draft -> Self-Correcting Remask)")
    print("Puzzle : 2-Hop Chained Conditional ('IF Box0 Even -> Swap 1<->2' + 'IF Box1 Even -> Swap 3<->4')")
    print("Model  : ONE single compact weight-shared block (trained at L=64 using parallel scan!)")
    print("=" * 86)

    torch.manual_seed(42)
    model = ThreePassRefinementWreath(vocab_size=64, d_model=64, d_g=6, d_v=24, num_vals=8)
    opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3)
    recent_acc = 0.125

    t0 = time.time()
    for step in range(1, 1601):
        lam, noise_std = compute_adaptive_control(step, recent_acc)
        dz = 0.05 if recent_acc >= 0.75 else 0.0

        toks, tgts = generate_two_hop_chained_puzzle(64, seq_len=64)
        pass_logits, m_loss = model(toks, noise_std=noise_std, deadzone=dz, dtype=torch.float32, chunk_snap=False)

        # Train Pass 1 on Box 0 skeleton, Pass 2 & Pass 3 on all boxes!
        loss_p1 = F.cross_entropy(pass_logits[0][:, 5:, 0].reshape(-1, 8), tgts[:, 5:, 0].reshape(-1))
        loss_p2 = F.cross_entropy(pass_logits[1][:, 5:].reshape(-1, 8), tgts[:, 5:].reshape(-1))
        loss_p3 = F.cross_entropy(pass_logits[2][:, 5:].reshape(-1, 8), tgts[:, 5:].reshape(-1))
        loss = 0.2 * loss_p1 + 0.4 * loss_p2 + 1.0 * loss_p3 + lam * m_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            b_acc = (pass_logits[2][:, -1, 3:].argmax(dim=-1) == tgts[:, -1, 3:]).float().mean().item()
            recent_acc = 0.90 * recent_acc + 0.10 * b_acc

    dt = time.time() - t0
    print(f"Trained 1,600 steps (all parallel-chunked!) in {dt:.1f}s")

    model.eval()
    eval_lengths = [64, 512, 2048, 8192]
    torch.manual_seed(555)
    eval_sets = {L: generate_two_hop_chained_puzzle(32, L) for L in eval_lengths}

    pass_names = [
        "Pass 1 (Skeleton Only - Swapper OFF)",
        "Pass 2 (Full Draft - 1st Swap Pass) ",
        "Pass 3 (Self-Correcting Remask)     ",
    ]
    with torch.no_grad():
        per_pass_results = {0: [], 1: [], 2: []}
        for L in eval_lengths:
            etoks, etgts = eval_sets[L]
            p_logits, _ = model(
                etoks, deadzone=0.05, dtype=torch.bfloat16, chunk_snap=True, conf_threshold=0.75
            )
            for p_idx in range(3):
                # Measure accuracy on Hop-2 Boxes 3 & 4 (hardest chained boxes!) and overall Boxes 1..4
                acc_hop2 = (p_logits[p_idx][:, -1, 3:].argmax(dim=-1) == etgts[:, -1, 3:]).float().mean().item()
                acc_all = (p_logits[p_idx][:, -1, 1:].argmax(dim=-1) == etgts[:, -1, 1:]).float().mean().item()
                per_pass_results[p_idx].append(f"L={L}: {acc_all*100:5.1f}% (Hop2: {acc_hop2*100:5.1f}%)")

        for p_idx in range(3):
            print(f"  {pass_names[p_idx]} -> " + " | ".join(per_pass_results[p_idx]))


if __name__ == "__main__":
    run_part_a_chunk_benchmark()
    run_part_b_three_pass_refinement()
