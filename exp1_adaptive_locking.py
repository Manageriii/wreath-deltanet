"""
Experiment #1: Solving Identity Collapse on Distractor-Heavy Sequences
======================================================================
In real text/code, ~80% of tokens are "filler" words that do NOT swap state,
making learning much slower than clean synthetic puzzles.

Compare 3 controllers for the {-1, 0, +1} magnet (`lam`) and exploration (`noise_std`):
  1. "no_magnet"    : Smooth ramp forever (lam = 0). Learns short L, drifts at long L!
  2. "blind_timer"  : Hardcoded timer (turns on magnet at step 150). Risks freezing too early!
  3. "adaptive_ours": YOUR design! Uses live `recent_accuracy` (0.0 to 1.0) to control
                      magnet strength `lam` and exploration `noise_std`.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# YOUR RESEARCH SANDBOX: We will fill in this function together!
# =====================================================================
def compute_adaptive_control(step: int, recent_accuracy: float) -> tuple[float, float]:
    """
    User-designed 3-stage adaptive controller:
      - Low (< 55%)    : No magnet (lam = 0.0), occasional bigger jumps (noise_std = 0.04)
      - Medium (55-85%): No magnet yet (lam = 0.0), moderate jumps (noise_std = 0.02)
      - High (>= 85%)  : Steadily exploring fine adjustments while ramping magnet up to 0.5!
    """
    if recent_accuracy < 0.55:
        return 0.0, 0.04
    elif recent_accuracy < 0.85:
        return 0.0, 0.02
    else:
        progress = min(1.0, (recent_accuracy - 0.85) / 0.13)
        lam = 0.5 * progress
        noise_std = 0.008 * (1.0 - 0.75 * progress)  # Steady gentle exploration
        return lam, noise_std


# =====================================================================
# Puzzle Generator: 5 Boxes + S_5 Swaps + Overwrites + 80% Filler Words
# =====================================================================
def generate_noisy_text_puzzle(batch_size: int, seq_len: int, num_vals: int = 8):
    """
    Tokens:
      0..3   : Swap adjacent boxes (0<->1, 1<->2, 2<->3, 3<->4)
      4..23  : 20 different "filler / chatter" words that should leave everything unchanged!
      24..63 : Write(box r, value v) -> puts `v` into `box r` (erasing old value)
    """
    num_filler = 20
    write_offset = 4 + num_filler
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # Initialize the 5 boxes in the first 5 tokens
    init_vals = torch.randint(0, num_vals, (batch_size, 5))
    for r in range(5):
        tokens[:, r] = write_offset + r * num_vals + init_vals[:, r]

    # Rest of sequence: 80% filler words, 14% box swaps, 6% box overwrites!
    u = torch.rand(batch_size, seq_len - 5)
    fillers = 4 + torch.randint(0, num_filler, (batch_size, seq_len - 5))
    swaps = torch.randint(0, 4, (batch_size, seq_len - 5))
    writes = write_offset + torch.randint(0, 5 * num_vals, (batch_size, seq_len - 5))

    rest = torch.where(u < 0.80, fillers, torch.where(u < 0.94, swaps, writes))
    tokens[:, 5:] = rest

    # Ground-truth box contents at every step
    boxes = torch.zeros(batch_size, 5, dtype=torch.long)
    targets = torch.zeros(batch_size, seq_len, 5, dtype=torch.long)
    for t in range(seq_len):
        tk = tokens[:, t]
        for g in range(4):
            m = tk == g
            if m.any():
                tmp = boxes[m, g].clone()
                boxes[m, g] = boxes[m, g + 1]
                boxes[m, g + 1] = tmp
        wm = tk >= write_offset
        if wm.any():
            wt = tk[wm] - write_offset
            boxes[wm, wt // num_vals] = wt % num_vals
        targets[:, t] = boxes
    return tokens, targets


# =====================================================================
# Chunk-Boundary Snap Helper: Locks numbers near {-1, 0, +1} to exact values
# =====================================================================
def snap_to_exact_signed_perm(U: torch.Tensor) -> torch.Tensor:
    B, d, _ = U.shape
    out = torch.zeros_like(U)
    work = U.abs().clone()
    signs = U.sign()
    ar = torch.arange(B, device=U.device)
    for _ in range(d):
        idx = work.view(B, -1).argmax(dim=-1)
        r, c = idx // d, idx % d
        out[ar, r, c] = signs[ar, r, c]
        work[ar, r, :] = -1.0
        work[ar, :, c] = -1.0
    return out


# =====================================================================
# Wreath-DeltaNet Layer (Group Reflection Frame + Delta-Rule Value Boxes)
# =====================================================================
class WreathDeltaNet(nn.Module):
    def __init__(self, vocab_size: int = 64, d_g: int = 6, d_v: int = 24, num_vals: int = 8):
        super().__init__()
        self.d_g = d_g
        self.d_v = d_v
        self.emb = nn.Embedding(vocab_size, 48)

        # Group reflection parameters (beta in [0, 2] allows flips/swaps!)
        self.to_hh_k = nn.Linear(48, 2 * d_g)
        self.to_hh_b = nn.Linear(48, 2)

        # Delta-Rule value memory parameters (beta in [0, 1] erases old & writes new!)
        self.to_k = nn.Linear(48, d_g)
        self.to_v = nn.Linear(48, d_v)
        self.to_beta = nn.Linear(48, 1)
        self.box_queries = nn.Parameter(torch.eye(d_g)[:5])
        self.readout = nn.Linear(d_v, num_vals)

    def forward(self, tokens, noise_std: float = 0.0, dtype=torch.float32, chunk_snap: bool = False):
        B, L = tokens.shape
        x = self.emb(tokens)

        raw_k = self.to_hh_k(x).view(B, L, 2, self.d_g)
        raw_b = self.to_hh_b(x)
        if self.training and noise_std > 0.0:
            raw_k = raw_k + torch.randn_like(raw_k) * noise_std
            raw_b = raw_b + torch.randn_like(raw_b) * noise_std

        hk = F.normalize(raw_k, dim=-1)
        hb = 2.0 * torch.sigmoid(raw_b)  # Range (0, 2): 0 = keep, 2 = mirror flip/swap!
        I = torch.eye(self.d_g, device=x.device).view(1, 1, self.d_g, self.d_g)
        H1 = I - hb[:, :, 0:1].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 0], hk[:, :, 0])
        H2 = I - hb[:, :, 1:2].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 1], hk[:, :, 1])
        G = torch.matmul(H2, H1)

        # Valley magnet loss: hits EXACT ZERO right at {-1, 0, +1} so weights never overflow!
        ortho_valley = (((G ** 2).sum(dim=-1) - 1.0) ** 2).mean()
        pm1_valley = (self.d_g - (G ** 4).sum(dim=(-2, -1))).mean() / self.d_g
        magnet_loss = ortho_valley + pm1_valley

        # Cumulative group frame U_t (snapped only once every 64-token chunk at eval!)
        Gc = G.to(dtype)
        Uc = torch.eye(self.d_g, device=x.device, dtype=dtype).unsqueeze(0).expand(B, self.d_g, self.d_g)
        U_list = []
        for t in range(L):
            Uc = torch.bmm(Gc[:, t], Uc)
            if chunk_snap and ((t + 1) % 64 == 0):
                Uc = snap_to_exact_signed_perm(Uc.float()).to(dtype)
            U_list.append(Uc)
        U = torch.stack(U_list, dim=1).float()

        # Delta-Rule memory boxes rotated by U_t^T
        k = F.normalize(self.to_k(x), dim=-1)
        v = self.to_v(x)
        beta = torch.sigmoid(self.to_beta(x))

        k_eff = torch.einsum("btij,bti->btj", U, k).to(dtype)
        q_eff = torch.einsum("btij,ri->btrj", U, F.normalize(self.box_queries, dim=-1)).to(dtype)

        S = torch.zeros(B, self.d_v, self.d_g, device=x.device, dtype=dtype)
        vc, bc = v.to(dtype), beta.to(dtype)
        outs = []
        for t in range(L):
            kt, vt, bt = k_eff[:, t], vc[:, t], bc[:, t, 0]
            old_val = torch.einsum("bvg,bg->bv", S, kt)
            S = S + torch.einsum("bv,bg->bvg", (vt - old_val) * bt.unsqueeze(-1), kt)
            outs.append(torch.einsum("bvg,brg->brv", S, q_eff[:, t]))

        logits = self.readout(torch.stack(outs, dim=1).float())
        return logits, magnet_loss, pm1_valley.item()


# =====================================================================
# Fair Head-to-Head Runner: Identical Seeds, Data, and BF16 Evaluation
# =====================================================================
def run_controller_comparison():
    controllers = [
        ("1. No Magnet (Smooth Only)", "no_magnet"),
        ("2. Early Blind Timer (Step 80)", "blind_timer"),
        ("3. Adaptive Rule (Ours)", "adaptive_ours"),
    ]
    eval_lengths = [48, 512, 2048, 8192]

    # Pre-generate fixed evaluation batches so all 3 models take the exact same test!
    torch.manual_seed(999)
    eval_sets = {L: generate_noisy_text_puzzle(32, L) for L in eval_lengths}

    print("=" * 80)
    print("EXPERIMENT #1: 80%-Filler Text + S_5 Box Swaps + Mutable Overwrites")
    print("Training at short length L=48 -> Testing zero-shot BF16 out to L=8,192 (170x)")
    print("=" * 80)

    for label, mode in controllers:
        torch.manual_seed(12)  # Same starting weights & training batches!
        model = WreathDeltaNet(vocab_size=64, d_g=6, d_v=24, num_vals=8)
        opt = torch.optim.AdamW(model.parameters(), lr=4e-3)

        recent_acc = 0.125
        t0 = time.time()

        for step in range(1, 1101):
            if mode == "no_magnet":
                lam, noise_std = 0.0, 0.0
            elif mode == "blind_timer":
                # Turns magnet on hard at step 80 before rare swaps are learned!
                lam = 0.0 if step < 80 else 0.8
                noise_std = 0.0
            elif mode == "adaptive_ours":
                lam, noise_std = compute_adaptive_control(step, recent_acc)

            toks, tgts = generate_noisy_text_puzzle(64, seq_len=48)
            logits, magnet_loss, valley_gap = model(toks, noise_std=noise_std)
            task_loss = F.cross_entropy(logits[:, 5:].reshape(-1, 8), tgts[:, 5:].reshape(-1))
            loss = task_loss + lam * magnet_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                batch_acc = (logits[:, -1].argmax(dim=-1) == tgts[:, -1]).float().mean().item()
                recent_acc = 0.90 * recent_acc + 0.10 * batch_acc

        dt = time.time() - t0
        model.eval()
        use_snap = (mode != "no_magnet")  # Only snapped if aligned to {-1, 0, +1}!
        scores = []
        with torch.no_grad():
            for L in eval_lengths:
                etoks, etgts = eval_sets[L]
                elogits, _, _ = model(etoks, dtype=torch.bfloat16, chunk_snap=use_snap)
                acc = (elogits[:, -1].argmax(dim=-1) == etgts[:, -1]).float().mean().item()
                scores.append(f"L={L}: {acc*100:5.1f}%")

        print(f"\n{label} ({dt:.1f}s | final valley_gap={valley_gap:.4f})")
        print("  BF16 Accuracy -> " + " | ".join(scores))


if __name__ == "__main__":
    run_controller_comparison()
