"""
Experiment #3: Conditional "If-Then" Reasoning & 2-Layer Specialization
=======================================================================
Rule added: `IF_BOX0_EVEN_SWAP_1_2` -> checks the current number stored in Box 0
(which may have been set hundreds or thousands of filler words ago!).
  - If Box 0 is EVEN (0, 2, 4, 6): Swap Box 1 <-> Box 2!
  - If Box 0 is ODD  (1, 3, 5, 7): Do nothing!

Tests 4 architectures (trained at L=48, evaluated zero-shot in BF16 out to L=8,192):
  1. 1-Layer Wreath-DeltaNet (Swapper can't see Box 0 yet!)
  2. 2-Layer Standard DeltaNet (No β=2 reflections in either layer)
  3. 2-Layer Hybrid (User's idea: Layer 1 = Value-Only [no swapper], Layer 2 = Wreath!)
  4. 2-Layer Full Wreath-DeltaNet (Both layers have Swappers + Dead-Zone + Adaptive Magnet)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_adaptive_locking import compute_adaptive_control, snap_to_exact_signed_perm
from exp2_deadzone_gate import apply_deadzone_gate


def generate_conditional_swap_puzzle(batch_size: int, seq_len: int, num_vals: int = 8):
    """
    Tokens (vocab = 65):
      0..2   : Unconditional swaps on Boxes 1..4: (1<->2), (2<->3), (3<->4)
      3      : CONDITIONAL SWAP: Swap (1<->2) ONLY IF Box 0 is currently EVEN!
      4..23  : 20 filler chatter words (do nothing)
      24..63 : Write(box r, val v) for r in 0..4, v in 0..7
    """
    num_filler = 20
    write_offset = 4 + num_filler
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # Initialize all 5 boxes (0..4) in the first 5 steps
    init_vals = torch.randint(0, num_vals, (batch_size, 5))
    for r in range(5):
        tokens[:, r] = write_offset + r * num_vals + init_vals[:, r]

    # Rest: 74% filler, 10% unconditional swaps (0..2), 8% conditional swap (3), 8% writes!
    u = torch.rand(batch_size, seq_len - 5)
    fillers = 4 + torch.randint(0, num_filler, (batch_size, seq_len - 5))
    uncond_swaps = torch.randint(0, 3, (batch_size, seq_len - 5))
    cond_swap = torch.full((batch_size, seq_len - 5), 3, dtype=torch.long)
    writes = write_offset + torch.randint(0, 5 * num_vals, (batch_size, seq_len - 5))

    rest = torch.where(
        u < 0.74,
        fillers,
        torch.where(u < 0.84, uncond_swaps, torch.where(u < 0.92, cond_swap, writes)),
    )
    tokens[:, 5:] = rest

    boxes = torch.zeros(batch_size, 5, dtype=torch.long)
    targets = torch.zeros(batch_size, seq_len, 5, dtype=torch.long)
    for t in range(seq_len):
        tk = tokens[:, t]
        # Unconditional swaps: g=0 swaps 1<->2, g=1 swaps 2<->3, g=2 swaps 3<->4
        for g in range(3):
            m = tk == g
            if m.any():
                b1, b2 = g + 1, g + 2
                tmp = boxes[m, b1].clone()
                boxes[m, b1] = boxes[m, b2]
                boxes[m, b2] = tmp
        # Conditional swap (token == 3): swap 1<->2 ONLY IF boxes[:, 0] is even!
        cm = (tk == 3) & ((boxes[:, 0] % 2) == 0)
        if cm.any():
            tmp = boxes[cm, 1].clone()
            boxes[cm, 1] = boxes[cm, 2]
            boxes[cm, 2] = tmp
        # Mutable writes to any box 0..4
        wm = tk >= write_offset
        if wm.any():
            wt = tk[wm] - write_offset
            boxes[wm, wt // num_vals] = wt % num_vals
        targets[:, t] = boxes
    return tokens, targets


class SequenceBlock(nn.Module):
    """
    One general-purpose sequence block (takes hidden stream [B, L, d_model] -> [B, L, d_model]).
    Can operate either WITH a Wreath Box-Swapper (`use_swapper=True`) or WITHOUT (`use_swapper=False`).
    """

    def __init__(self, d_model: int = 64, d_g: int = 6, d_v: int = 24, use_swapper: bool = True):
        super().__init__()
        self.d_g = d_g
        self.d_v = d_v
        self.use_swapper = use_swapper

        self.norm1 = nn.LayerNorm(d_model)
        if use_swapper:
            self.to_hh_k = nn.Linear(d_model, 2 * d_g)
            self.to_hh_b = nn.Linear(d_model, 2)

        self.to_q = nn.Linear(d_model, d_g)
        self.to_k = nn.Linear(d_model, d_g)
        self.to_v = nn.Linear(d_model, d_v)
        self.to_beta = nn.Linear(d_model, 1)
        self.box_probes = nn.Parameter(torch.eye(d_g)[:5])
        self.proj_out = nn.Linear(d_v, d_model)

        # Tiny 2-layer MLP (just like every Transformer block) so retrieved values can compute rules like even/odd!
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        h: torch.Tensor,
        noise_std: float = 0.0,
        deadzone: float = 0.0,
        dtype=torch.float32,
        chunk_snap: bool = False,
        return_probes_final_only: bool = False,
    ):
        B, L, _ = h.shape
        x = self.norm1(h)

        if self.use_swapper:
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
        else:
            magnet_loss = torch.tensor(0.0, device=h.device)
            Gc = None

        q_tok = F.normalize(self.to_q(x), dim=-1)
        k = F.normalize(self.to_k(x), dim=-1)
        v = self.to_v(x).to(dtype)
        beta = apply_deadzone_gate(self.to_beta(x), zone=deadzone, active=(deadzone > 0.0)).to(dtype)
        probes_norm = F.normalize(self.box_probes, dim=-1)

        Uc = torch.eye(self.d_g, device=x.device, dtype=dtype).unsqueeze(0).expand(B, self.d_g, self.d_g)
        S = torch.zeros(B, self.d_v, self.d_g, device=x.device, dtype=dtype)

        stream_outs = []
        probe_outs = []

        for t in range(L):
            if self.use_swapper:
                Uc = torch.bmm(Gc[:, t], Uc)
                if chunk_snap and ((t + 1) % 64 == 0):
                    Uc = snap_to_exact_signed_perm(Uc.float()).to(dtype)
                kt_eff = torch.einsum("bij,bi->bj", Uc.float(), k[:, t]).to(dtype)
                qt_eff = torch.einsum("bij,bi->bj", Uc.float(), q_tok[:, t]).to(dtype)
            else:
                kt_eff = k[:, t].to(dtype)
                qt_eff = q_tok[:, t].to(dtype)

            bt = beta[:, t, 0]
            old_val = torch.einsum("bvg,bg->bv", S, kt_eff)
            S = S + torch.einsum("bv,bg->bvg", (v[:, t] - old_val) * bt.unsqueeze(-1), kt_eff)

            # Readout passed up to the next layer at every token!
            stream_outs.append(torch.einsum("bvg,bg->bv", S, qt_eff))

            # Probe all 5 boxes for final answer readout
            if not return_probes_final_only or (t == L - 1):
                if self.use_swapper:
                    pt_eff = torch.einsum("bij,ri->brj", Uc.float(), probes_norm).to(dtype)
                else:
                    pt_eff = probes_norm.unsqueeze(0).expand(B, -1, -1).to(dtype)
                probe_outs.append(torch.einsum("bvg,brg->brv", S, pt_eff))

        readout_seq = torch.stack(stream_outs, dim=1).float()
        h_next = h + self.proj_out(readout_seq)
        h_next = h_next + self.mlp(self.norm2(h_next))

        probes_seq = torch.stack(probe_outs, dim=1).float()
        return h_next, probes_seq, magnet_loss


class MultiLayerWreathModel(nn.Module):
    def __init__(self, layer_swapper_flags: list[bool], vocab_size: int = 65, d_model: int = 64, num_vals: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [SequenceBlock(d_model=d_model, d_g=6, d_v=24, use_swapper=flag) for flag in layer_swapper_flags]
        )
        self.classifier = nn.Linear(24, num_vals)

    def forward(
        self,
        tokens: torch.Tensor,
        noise_std: float = 0.0,
        deadzone: float = 0.0,
        dtype=torch.float32,
        chunk_snap: bool = False,
        final_only: bool = False,
    ):
        h = self.emb(tokens)
        total_magnet = torch.tensor(0.0, device=tokens.device)
        probes = None
        for i, blk in enumerate(self.blocks):
            is_last = i == (len(self.blocks) - 1)
            h, probes, m_loss = blk(
                h,
                noise_std=noise_std,
                deadzone=deadzone,
                dtype=dtype,
                chunk_snap=chunk_snap,
                return_probes_final_only=(final_only if is_last else True),
            )
            total_magnet = total_magnet + m_loss
        return self.classifier(probes), total_magnet


def run_conditional_experiment():
    architectures = [
        ("1. 1-Layer Wreath-DeltaNet", [True]),
        ("2. 2-Layer Standard DeltaNet (No Swappers)", [False, False]),
        ("3. 2-Layer Hybrid (L1: No Swapper, L2: Swapper)", [False, True]),
        ("4. 2-Layer Full Wreath-DeltaNet (Both Swappers)", [True, True]),
    ]
    eval_lengths = [48, 512, 2048, 8192]

    torch.manual_seed(777)
    eval_sets = {L: generate_conditional_swap_puzzle(32, L) for L in eval_lengths}

    print("=" * 84)
    print("EXPERIMENT #3: Conditional 'IF Box 0 is Even -> Swap Box 1<->2' (Trained @ L=48)")
    print("=" * 84)

    for label, flags in architectures:
        torch.manual_seed(21)
        model = MultiLayerWreathModel(layer_swapper_flags=flags)
        opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3)
        recent_acc = 0.125
        t0 = time.time()

        for step in range(1, 1501):
            lam, noise_std = compute_adaptive_control(step, recent_acc)
            dz = 0.05 if recent_acc >= 0.75 else 0.0

            toks, tgts = generate_conditional_swap_puzzle(64, seq_len=48)
            logits, magnet_loss = model(toks, noise_std=noise_std, deadzone=dz)
            task_loss = F.cross_entropy(logits[:, 5:].reshape(-1, 8), tgts[:, 5:].reshape(-1))
            loss = task_loss + lam * magnet_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                # Track accuracy specifically on Boxes 1..4 (where swaps & conditional swaps act!)
                batch_acc = (logits[:, -1, 1:].argmax(dim=-1) == tgts[:, -1, 1:]).float().mean().item()
                recent_acc = 0.90 * recent_acc + 0.10 * batch_acc

        dt = time.time() - t0
        model.eval()
        scores = []
        with torch.no_grad():
            for L in eval_lengths:
                etoks, etgts = eval_sets[L]
                elogits, _ = model(
                    etoks,
                    deadzone=0.05,
                    dtype=torch.bfloat16,
                    chunk_snap=True,
                    final_only=True,
                )
                acc_swapped = (elogits[:, 0, 1:].argmax(dim=-1) == etgts[:, -1, 1:]).float().mean().item()
                scores.append(f"L={L}: {acc_swapped*100:5.1f}%")

            # Inspect how active each layer's Box-Swapper is on a test batch!
            sample_toks, _ = eval_sets[48]
            h_probe = model.emb(sample_toks)
            swap_stats = []
            eye6 = torch.eye(6).unsqueeze(0).unsqueeze(0)
            for idx, blk in enumerate(model.blocks):
                if blk.use_swapper:
                    x_n = blk.norm1(h_probe)
                    hk = F.normalize(blk.to_hh_k(x_n).view(32, 48, 2, 6), dim=-1)
                    hb = 2.0 * torch.sigmoid(blk.to_hh_b(x_n))
                    I = torch.eye(6).view(1, 1, 6, 6)
                    H1 = I - hb[:, :, 0:1].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 0], hk[:, :, 0])
                    H2 = I - hb[:, :, 1:2].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 1], hk[:, :, 1])
                    G_snap = snap_to_exact_signed_perm(torch.matmul(H2, H1).reshape(-1, 6, 6)).view(32, 48, 6, 6)
                    pct_swapping = (~(G_snap == eye6).all(dim=-1).all(dim=-1)).float().mean().item() * 100.0
                    swap_stats.append(f"L{idx+1} swaps on {pct_swapping:.1f}% of tokens")
                else:
                    swap_stats.append(f"L{idx+1} swapper OFF (0.0%)")
                h_probe, _, _ = blk(h_probe, deadzone=0.05, chunk_snap=True, return_probes_final_only=True)

        print(f"\n{label} ({dt:.1f}s) [{' | '.join(swap_stats)}]")
        print("  BF16 Accuracy (Boxes 1-4) -> " + " | ".join(scores))


if __name__ == "__main__":
    run_conditional_experiment()
