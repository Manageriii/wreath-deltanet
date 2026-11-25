"""
Experiment #2: Plugging the Filler-Word "Leaky Faucet" with a Dead-Zone Gate
============================================================================
Tests extreme zero-shot BF16 length extrapolation from L_train = 48 all the way
to L_eval = 32,768 (682x longer -> ~26,200 filler chatter words per sequence!):
  1. Standard Smooth Only (No Magnet, No Dead-Zone)
  2. Exp #1 Winner: Adaptive Magnet (Leaky Sigmoid Write Knob)
  3. Exp #2 Ours  : Adaptive Magnet + Dead-Zone Write Knob (< 5% -> exact 0%)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_adaptive_locking import (
    compute_adaptive_control,
    generate_noisy_text_puzzle,
    snap_to_exact_signed_perm,
)


def apply_deadzone_gate(raw_beta: torch.Tensor, zone: float = 0.05, active: bool = True) -> torch.Tensor:
    """
    Turns a leaky (0%, 100%) sigmoid knob into a clean Dead-Zone knob:
      - Below `zone` (e.g., < 5%): clamped to EXACT 0.0 (zero leak, zero BF16 rounding!)
      - Above `1 - zone` (> 95%) : clamped to EXACT 1.0 (clean 100% overwrite!)
      - In between               : smooth ramp so learning still flows cleanly.
    """
    b = torch.sigmoid(raw_beta)
    if not active or zone <= 0.0:
        return b
    stretched = (b - zone) / (1.0 - 2.0 * zone)
    return torch.clamp(stretched, 0.0, 1.0)


class WreathDeltaNetV2(nn.Module):
    def __init__(self, vocab_size: int = 64, d_g: int = 6, d_v: int = 24, num_vals: int = 8):
        super().__init__()
        self.d_g = d_g
        self.d_v = d_v
        self.emb = nn.Embedding(vocab_size, 48)

        self.to_hh_k = nn.Linear(48, 2 * d_g)
        self.to_hh_b = nn.Linear(48, 2)

        self.to_k = nn.Linear(48, d_g)
        self.to_v = nn.Linear(48, d_v)
        self.to_beta = nn.Linear(48, 1)
        self.box_queries = nn.Parameter(torch.eye(d_g)[:5])
        self.readout = nn.Linear(d_v, num_vals)

    def forward(
        self,
        tokens: torch.Tensor,
        noise_std: float = 0.0,
        deadzone: float = 0.0,
        dtype=torch.float32,
        chunk_snap: bool = False,
        final_only: bool = False,
    ):
        B, L = tokens.shape
        x = self.emb(tokens)

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

        Gc = G.to(dtype)
        Uc = torch.eye(self.d_g, device=x.device, dtype=dtype).unsqueeze(0).expand(B, self.d_g, self.d_g)

        k = F.normalize(self.to_k(x), dim=-1)
        v = self.to_v(x).to(dtype)
        beta = apply_deadzone_gate(self.to_beta(x), zone=deadzone, active=(deadzone > 0.0)).to(dtype)
        q_norm = F.normalize(self.box_queries, dim=-1)

        S = torch.zeros(B, self.d_v, self.d_g, device=x.device, dtype=dtype)
        outs = []

        for t in range(L):
            Uc = torch.bmm(Gc[:, t], Uc)
            if chunk_snap and ((t + 1) % 64 == 0):
                Uc = snap_to_exact_signed_perm(Uc.float()).to(dtype)

            bt = beta[:, t, 0]  # [B]
            # Notice: when deadzone sets bt == 0, S is untouched (zero leak & zero BF16 rounding)!
            kt_eff = torch.einsum("bij,bi->bj", Uc.float(), k[:, t]).to(dtype)
            old_val = torch.einsum("bvg,bg->bv", S, kt_eff)
            S = S + torch.einsum("bv,bg->bvg", (v[:, t] - old_val) * bt.unsqueeze(-1), kt_eff)

            if not final_only or (t == L - 1):
                qt_eff = torch.einsum("bij,ri->brj", Uc.float(), q_norm).to(dtype)
                outs.append(torch.einsum("bvg,brg->brv", S, qt_eff))

        logits = self.readout(torch.stack(outs, dim=1).float())
        return logits, magnet_loss


def run_deadzone_experiment():
    configs = [
        ("1. Smooth Baseline (No Magnet, Leaky Knob)", False, False),
        ("2. Exp #1: Adaptive Magnet + Leaky Knob", True, False),
        ("3. Exp #2: Adaptive Magnet + Dead-Zone Knob (Ours)", True, True),
    ]
    eval_lengths = [48, 2048, 8192, 32768]

    torch.manual_seed(999)
    eval_sets = {L: generate_noisy_text_puzzle(32, L) for L in eval_lengths}

    print("=" * 82)
    print("EXPERIMENT #2: Testing the Dead-Zone Write Knob out to L = 32,768 (682x in BF16!)")
    print("=" * 82)

    for label, use_magnet, use_deadzone in configs:
        torch.manual_seed(12)
        model = WreathDeltaNetV2(vocab_size=64, d_g=6, d_v=24, num_vals=8)
        opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
        recent_acc = 0.125
        t0 = time.time()

        for step in range(1, 1101):
            if use_magnet:
                lam, noise_std = compute_adaptive_control(step, recent_acc)
            else:
                lam, noise_std = 0.0, 0.0

            # Activate the 5% dead-zone once the model passes 75% accuracy ("after some point"!)
            dz = 0.05 if (use_deadzone and recent_acc >= 0.75) else 0.0

            toks, tgts = generate_noisy_text_puzzle(64, seq_len=48)
            logits, magnet_loss = model(toks, noise_std=noise_std, deadzone=dz)
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
        eval_dz = 0.05 if use_deadzone else 0.0
        scores = []
        with torch.no_grad():
            for L in eval_lengths:
                etoks, etgts = eval_sets[L]
                elogits, _ = model(
                    etoks,
                    deadzone=eval_dz,
                    dtype=torch.bfloat16,
                    chunk_snap=use_magnet,
                    final_only=True,
                )
                acc = (elogits[:, 0].argmax(dim=-1) == etgts[:, -1]).float().mean().item()
                scores.append(f"L={L}: {acc*100:5.1f}%")

        print(f"\n{label} ({dt:.1f}s)")
        print("  BF16 Accuracy -> " + " | ".join(scores))


if __name__ == "__main__":
    run_deadzone_experiment()
