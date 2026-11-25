"""
Experiment #4B (Try #2): User's Clean Channel Split + Confidence-Anchored 3-Pass Refinement
===========================================================================================
Implements our two exact fixes—with ZERO hardcoded box indices in the loss:
  1. Worker Input Split:
     - Worker 2 (Value-Writer) ONLY sees clean original token embeddings (`base_h`).
     - Worker 1 (Box-Swapper) sees `base_h + confidence_filtered_draft`.
  2. Confidence-Anchored Handoff (applied uniformly between passes):
     - High-confidence boxes (>= 75%) lock as crisp anchors.
     - Uncertain boxes (< 75%) are scaled by their confidence score (0%-100%) instead of
       resetting back to Pass 1.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_adaptive_locking import compute_adaptive_control, snap_to_exact_signed_perm
from exp2_deadzone_gate import apply_deadzone_gate
from exp4_parallel_refinement import (
    generate_two_hop_chained_puzzle,
    parallel_chunked_group_scan,
)


def confidence_anchor_filter(box_feats: torch.Tensor, logits: torch.Tensor, threshold: float = 0.75) -> torch.Tensor:
    """
    User's Pass-Handoff Scoring Rule (differentiable!):
      - Compute confidence = max class probability in [0.125, 1.0] for each box.
      - Boxes with confidence >= 75% lock at full 1.0 weight (fixed anchor strength).
      - Uncertain boxes (< 75%) are scaled down by their confidence score (0%-75%)!
    """
    probs = F.softmax(logits, dim=-1)
    conf = probs.max(dim=-1, keepdim=True).values  # [B, L, 5, 1]
    target_scale = torch.where(conf >= threshold, torch.ones_like(conf), conf)
    scale = target_scale - conf.detach() + conf  # Keeps learning wire intact!
    return box_feats * scale


class CleanThreePassWreath(nn.Module):
    """
    ONE SINGLE COMPACT WEIGHT-SHARED MODEL running 3 refinement passes:
      Pass 1 (Skeleton, Swapper OFF) -> Pass 2 (Draft, Swapper ON) -> Pass 3 (Refined, Swapper ON)
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 64, d_g: int = 6, d_v: int = 24, num_vals: int = 8):
        super().__init__()
        self.d_g = d_g
        self.d_v = d_v

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Worker 1 (Box-Swapper) ONLY: nonlinearly combines current word + confidence-anchored box draft!
        self.swapper_fusion = nn.Sequential(
            nn.LayerNorm(d_model + 5 * d_v),
            nn.Linear(d_model + 5 * d_v, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
        )
        self.to_hh_k = nn.Linear(d_model, 2 * d_g)
        self.to_hh_b = nn.Linear(d_model, 2)

        # Worker 2 (Value-Writer) ONLY: receives ONLY clean `tok_emb` (zero draft pollution!)
        self.writer_norm = nn.LayerNorm(d_model)
        self.to_k = nn.Linear(d_model, d_g)
        self.to_v = nn.Linear(d_model, d_v)
        self.to_beta = nn.Linear(d_model, 1)
        self.box_queries = nn.Parameter(torch.eye(d_g)[:5])
        self.readout = nn.Linear(d_v, num_vals)

    def run_pass(
        self,
        base_h: torch.Tensor,
        prev_draft_ctx: torch.Tensor | None,
        use_swapper: bool,
        noise_std: float = 0.0,
        deadzone: float = 0.0,
        dtype=torch.float32,
        chunk_snap: bool = False,
        use_parallel_eval: bool = False,
    ):
        B, L, _ = base_h.shape

        # WORKER 2 inputs: 100% clean original words only!
        w_in = self.writer_norm(base_h)
        k = F.normalize(self.to_k(w_in), dim=-1)
        v = self.to_v(w_in)
        beta = apply_deadzone_gate(self.to_beta(w_in), zone=deadzone, active=(deadzone > 0.0))
        q_norm = F.normalize(self.box_queries, dim=-1)

        # WORKER 1 inputs: nonlinearly fuses clean word + previous pass's confidence-anchored box draft!
        if use_swapper:
            fused = torch.cat([base_h, prev_draft_ctx.reshape(B, L, -1)], dim=-1)
            s_in = self.swapper_fusion(fused)
            raw_k = self.to_hh_k(s_in).view(B, L, 2, self.d_g)
            raw_b = self.to_hh_b(s_in)
            if self.training and noise_std > 0.0:
                raw_k = raw_k + torch.randn_like(raw_k) * noise_std
                raw_b = raw_b + torch.randn_like(raw_b) * noise_std

            hk = F.normalize(raw_k, dim=-1)
            hb = 2.0 * torch.sigmoid(raw_b)
            I = torch.eye(self.d_g, device=base_h.device).view(1, 1, self.d_g, self.d_g)
            H1 = I - hb[:, :, 0:1].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 0], hk[:, :, 0])
            H2 = I - hb[:, :, 1:2].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 1], hk[:, :, 1])
            G = torch.matmul(H2, H1)

            ortho_valley = (((G ** 2).sum(dim=-1) - 1.0) ** 2).mean()
            pm1_valley = (self.d_g - (G ** 4).sum(dim=(-2, -1))).mean() / self.d_g
            magnet_loss = ortho_valley + pm1_valley

            if use_parallel_eval:
                U = parallel_chunked_group_scan(G, chunk_size=64, dtype=dtype, snap=chunk_snap)
            else:
                Gc = G.to(dtype)
                Uc = torch.eye(self.d_g, device=base_h.device, dtype=dtype).unsqueeze(0).expand(B, self.d_g, self.d_g)
                U_list = []
                for t in range(L):
                    Uc = torch.bmm(Gc[:, t], Uc)
                    if chunk_snap and ((t + 1) % 64 == 0):
                        Uc = snap_to_exact_signed_perm(Uc.float()).to(dtype)
                    U_list.append(Uc)
                U = torch.stack(U_list, dim=1).float()
        else:
            magnet_loss = torch.tensor(0.0, device=base_h.device)
            U = torch.eye(self.d_g, device=base_h.device).view(1, 1, self.d_g, self.d_g).expand(B, L, self.d_g, self.d_g)

        k_eff = torch.einsum("btij,bti->btj", U, k).to(dtype)
        q_eff = torch.einsum("btij,ri->btrj", U, q_norm).to(dtype)
        vc, bc = v.to(dtype), beta.to(dtype)

        S = torch.zeros(B, self.d_v, self.d_g, device=base_h.device, dtype=dtype)
        outs = []
        for t in range(L):
            kt, vt, bt = k_eff[:, t], vc[:, t], bc[:, t, 0]
            old_val = torch.einsum("bvg,bg->bv", S, kt)
            S = S + torch.einsum("bv,bg->bvg", (vt - old_val) * bt.unsqueeze(-1), kt)
            outs.append(torch.einsum("bvg,brg->brv", S, q_eff[:, t]))

        box_feats = torch.stack(outs, dim=1).float()  # [B, L, 5, d_v]
        logits = self.readout(box_feats)              # [B, L, 5, 8]
        return box_feats, logits, magnet_loss

    def forward(
        self,
        tokens: torch.Tensor,
        noise_std: float = 0.0,
        deadzone: float = 0.0,
        dtype=torch.float32,
        chunk_snap: bool = False,
        use_parallel_eval: bool = False,
    ):
        B, L = tokens.shape
        base_h = self.tok_emb(tokens)
        zero_pad = torch.zeros(B, 1, 5, self.d_v, device=tokens.device)

        # PASS 1: Skeleton (Swapper OFF)
        f1, logits_p1, _ = self.run_pass(
            base_h, None, use_swapper=False, deadzone=deadzone, dtype=dtype, use_parallel_eval=use_parallel_eval
        )
        # Filter Pass 1 with User's Confidence Rule (locks >= 75%, scales < 75%)
        g1 = confidence_anchor_filter(f1, logits_p1, threshold=0.75)
        ctx_1_to_2 = torch.cat([zero_pad, g1[:, :-1]], dim=1)

        # PASS 2: Full Draft (Swapper ON, sees Pass 1's anchored Box 0 -> resolves Box 1 & 2!)
        f2, logits_p2, m_loss2 = self.run_pass(
            base_h,
            ctx_1_to_2,
            use_swapper=True,
            noise_std=noise_std,
            deadzone=deadzone,
            dtype=dtype,
            chunk_snap=chunk_snap,
            use_parallel_eval=use_parallel_eval,
        )
        # Filter Pass 2 with User's Confidence Rule (locks Box 0 & Box 1 >= 75%, scales Box 3 & 4!)
        g2 = confidence_anchor_filter(f2, logits_p2, threshold=0.75)
        ctx_2_to_3 = torch.cat([zero_pad, g2[:, :-1]], dim=1)

        # PASS 3: Self-Correcting Refinement (Swapper ON, sees locked Box 0 & Box 1 -> resolves Box 3 & 4!)
        _, logits_p3, m_loss3 = self.run_pass(
            base_h,
            ctx_2_to_3,
            use_swapper=True,
            noise_std=noise_std,
            deadzone=deadzone,
            dtype=dtype,
            chunk_snap=chunk_snap,
            use_parallel_eval=use_parallel_eval,
        )
        return [logits_p1, logits_p2, logits_p3], 0.5 * (m_loss2 + m_loss3)


def run_v2_experiment():
    print("=" * 88)
    print("EXPERIMENT #4B (TRY #2): Clean Worker Split + Confidence-Anchored 3-Pass Refinement")
    print("Puzzle : 2-Hop Chained ('IF Box0 Even -> Swap 1<->2' AND 'IF Box1 Even -> Swap 3<->4')")
    print("Weights: ONE single compact weight-shared model (ZERO hardcoded box indices in loss!)")
    print("=" * 88)

    torch.manual_seed(42)
    model = CleanThreePassWreath(vocab_size=64, d_model=64, d_g=6, d_v=24, num_vals=8)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
    recent_acc = 0.125

    t0 = time.time()
    for step in range(1, 1501):
        lam, noise_std = compute_adaptive_control(step, recent_acc)
        dz = 0.05 if recent_acc >= 0.75 else 0.0

        toks, tgts = generate_two_hop_chained_puzzle(64, seq_len=48)
        p_logits, m_loss = model(toks, noise_std=noise_std, deadzone=dz)

        # Supervise the refined Pass 3 output (+ gentle Pass 2 draft guidance)
        l2 = F.cross_entropy(p_logits[1][:, 5:].reshape(-1, 8), tgts[:, 5:].reshape(-1))
        l3 = F.cross_entropy(p_logits[2][:, 5:].reshape(-1, 8), tgts[:, 5:].reshape(-1))
        loss = 0.2 * l2 + 1.0 * l3 + lam * m_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            # Monitor final Pass-3 accuracy across all swapped boxes 1..4
            b_acc = (p_logits[2][:, -1, 1:].argmax(dim=-1) == tgts[:, -1, 1:]).float().mean().item()
            recent_acc = 0.90 * recent_acc + 0.10 * b_acc

    dt = time.time() - t0
    print(f"Trained 1,500 steps @ L=48 in {dt:.1f}s (final running Pass-3 acc = {recent_acc*100:.1f}%)\n")

    model.eval()
    eval_lengths = [48, 512, 2048, 8192]
    torch.manual_seed(555)
    eval_sets = {L: generate_two_hop_chained_puzzle(32, L) for L in eval_lengths}

    pass_labels = [
        "Pass 1 (Skeleton - Swapper OFF)  ",
        "Pass 2 (Hop-1 Draft)             ",
        "Pass 3 (Confidence-Locked Refine)",
    ]
    with torch.no_grad():
        rows = {0: [], 1: [], 2: []}
        for L in eval_lengths:
            etoks, etgts = eval_sets[L]
            p_logits, _ = model(
                etoks, deadzone=0.05, dtype=torch.bfloat16, chunk_snap=True, use_parallel_eval=True
            )
            for p_i in range(3):
                acc_hop1 = (p_logits[p_i][:, -1, 1:3].argmax(dim=-1) == etgts[:, -1, 1:3]).float().mean().item()
                acc_hop2 = (p_logits[p_i][:, -1, 3:5].argmax(dim=-1) == etgts[:, -1, 3:5]).float().mean().item()
                acc_all = (p_logits[p_i][:, -1, 1:5].argmax(dim=-1) == etgts[:, -1, 1:5]).float().mean().item()
                rows[p_i].append(f"L={L}: {acc_all*100:5.1f}% [Hop2:{acc_hop2*100:5.1f}%]")

        for p_i in range(3):
            print(f"  {pass_labels[p_i]} -> " + " | ".join(rows[p_i]))


if __name__ == "__main__":
    run_v2_experiment()
