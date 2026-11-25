"""
Experiment #5 (Option A): 8-Bit FP8 Stress Test & Spontaneous "Two-Gear" Clock Discovery
========================================================================================
Part 5A: Sweeps chunk sizes C = [Unsnapped, 256, 64, 16, 8] under true 8-bit FP8 (E4M3)
         rounding noise (3 mantissa bits -> 16x coarser than BF16!) out to L = 16,384.
Part 5B: Trains a model with ONLY d_g = 8 boxes (< 12!) on a 12-Hour Clock + Resets +
         75% Filler Words (StateBench Tier 2 + Tier 3), tests zero-shot FP8 out to
         L = 16,384, and inspects the exact {-1, 0, +1} gears the AI invented!
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_adaptive_locking import compute_adaptive_control, snap_to_exact_signed_perm
from exp2_deadzone_gate import apply_deadzone_gate


# =====================================================================
# Hardware-Accurate 8-Bit FP8 (E4M3: 1 sign, 4 exponent, 3 mantissa bits)
# =====================================================================
def quantize_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Simulates IEEE/NVIDIA FP8 E4M3 round-to-nearest-even (step = 2^-3 = 12.5% per binade!)."""
    out = torch.zeros_like(x)
    mask = x.abs() > 1e-12
    if not mask.any():
        return out
    ax = x[mask].abs().clamp(max=448.0)
    exp = torch.floor(torch.log2(ax)).clamp(min=-6.0)
    step = torch.pow(2.0, exp - 3.0)
    out[mask] = torch.sign(x[mask]) * torch.round(ax / step) * step
    return out


def fp8_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Quantizes inputs AND output to FP8 E4M3 to stress-test chunk accumulation drift."""
    return quantize_fp8_e4m3(torch.matmul(quantize_fp8_e4m3(A), quantize_fp8_e4m3(B)))


def fp8_chunked_scan_generic_basis(ks_seq: torch.Tensor, active_seq: torch.Tensor, chunk_size: int | None, V_rot: torch.Tensor):
    """
    Runs S_5 reflections in a generic dense rotated basis V_rot under 8-bit FP8 arithmetic,
    snapping back to the discrete lattice every `chunk_size` steps (or never if None).
    """
    B, L, d = ks_seq.shape
    U = torch.eye(d).unsqueeze(0).repeat(B, 1, 1)
    I = torch.eye(d).unsqueeze(0)

    for t in range(L):
        k_q = quantize_fp8_e4m3(ks_seq[:, t])
        act = active_seq[:, t].view(B, 1, 1)
        H_q = quantize_fp8_e4m3(I - 2.0 * act * torch.einsum("bi,bj->bij", k_q, k_q))
        U = fp8_matmul(H_q, U)
        if (chunk_size is not None) and ((t + 1) % chunk_size == 0):
            # Transform to canonical basis, snap to exact {-1, 0, +1}, rotate back!
            U_can = V_rot.T @ U @ V_rot
            U_snap = snap_to_exact_signed_perm(U_can)
            U = quantize_fp8_e4m3(V_rot @ U_snap @ V_rot.T)
    return U


# =====================================================================
# PART 5A: Sweeping C = [None, 256, 64, 16, 8] under 8-Bit FP8 Noise!
# =====================================================================
def run_part_5a_fp8_chunk_sweep():
    print("=" * 88)
    print("PART 5A: 8-Bit FP8 (E4M3) Chunk-Size Survival Sweep out to L = 16,384")
    print("Testing: Unsnapped vs C=256 vs C=64 vs YOUR C=16 and C=8!")
    print("=" * 88)

    torch.manual_seed(42)
    d = 6
    V_rot, _ = torch.linalg.qr(torch.randn(d, d))

    # Build S_5 generators in rotated dense coordinates
    ks_table = []
    for i in range(4):
        v = torch.zeros(d)
        v[i] = 1.0 / math.sqrt(2.0)
        v[i + 1] = -1.0 / math.sqrt(2.0)
        ks_table.append(V_rot @ v)
    ks_table.append(torch.zeros(d))
    ks_table = torch.stack(ks_table)

    B = 128
    eval_lengths = [64, 256, 1024, 4096, 16384]
    max_L = max(eval_lengths)
    gen_ids = torch.randint(0, 5, (B, max_L))

    # Ground-truth exact trajectory
    U_true_at = {}
    U_t = torch.eye(d).unsqueeze(0).repeat(B, 1, 1)
    for t in range(max_L):
        g = gen_ids[:, t]
        k = ks_table[g]
        act = (g < 4).float().view(B, 1, 1)
        H = torch.eye(d).unsqueeze(0) - 2.0 * act * torch.einsum("bi,bj->bij", k, k)
        U_t = torch.bmm(H, U_t)
        if (t + 1) in eval_lengths:
            U_true_at[t + 1] = snap_to_exact_signed_perm(V_rot.T @ U_t @ V_rot)

    for label, C in [
        ("FP8 Unsnapped (No Snap) ", None),
        ("FP8 Chunk C = 256       ", 256),
        ("FP8 Chunk C =  64 (BF16)", 64),
        ("FP8 Chunk C =  16       ", 16),
        ("FP8 Chunk C =   8 (Edge)", 8),
        ("FP8 Chunk C =   4 (Ours)", 4),
    ]:
        U_est = torch.eye(d).unsqueeze(0).repeat(B, 1, 1)
        I = torch.eye(d).unsqueeze(0)
        res = []
        for t in range(max_L):
            g = gen_ids[:, t]
            k_q = quantize_fp8_e4m3(ks_table[g])
            act = (g < 4).float().view(B, 1, 1)
            H_q = quantize_fp8_e4m3(I - 2.0 * act * torch.einsum("bi,bj->bij", k_q, k_q))
            U_est = fp8_matmul(H_q, U_est)

            if (C is not None) and ((t + 1) % C == 0):
                U_can = V_rot.T @ U_est @ V_rot
                U_snap = snap_to_exact_signed_perm(U_can)
                U_est = quantize_fp8_e4m3(V_rot @ U_snap @ V_rot.T)

            if (t + 1) in eval_lengths:
                pred_can = snap_to_exact_signed_perm(V_rot.T @ U_est @ V_rot)
                acc = (pred_can == U_true_at[t + 1]).all(dim=-1).all(dim=-1).float().mean().item()
                res.append(f"L={t+1}:{acc*100:5.1f}%")
        print(f"  {label} -> " + " | ".join(res))


# =====================================================================
# PART 5B: 12-Hour Clock ("Two-Gear Discovery") Inside ONLY 8 Boxes!
# =====================================================================
def generate_12hr_clock_puzzle(batch_size: int, seq_len: int):
    """
    12-Hour Clock Puzzle (StateBench Tier 2 Mutable Reset + Tier 3 Soluble Clock):
      - Token 0     : +1 hour  (mod 12)
      - Token 1     : +2 hours (mod 12)
      - Token 2     : +5 hours (mod 12)
      - Token 3     : -1 hour  (+11 mod 12)
      - Token 4..23 : 20 filler chatter words (+0 hours)
      - Token 24..35: Reset Clock = h (sets clock directly to h in 0..11!)
    """
    reset_offset = 24
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    # Step 0 sets a random starting hour 0..11
    tokens[:, 0] = reset_offset + torch.randint(0, 12, (batch_size,))

    u = torch.rand(batch_size, seq_len - 1)
    fillers = 4 + torch.randint(0, 20, (batch_size, seq_len - 1))
    shifts = torch.randint(0, 4, (batch_size, seq_len - 1))
    resets = reset_offset + torch.randint(0, 12, (batch_size, seq_len - 1))

    # 75% filler chatter, 20% clock ticks (+1, +2, +5, -1), 5% clock resets!
    tokens[:, 1:] = torch.where(u < 0.75, fillers, torch.where(u < 0.95, shifts, resets))

    deltas = {0: 1, 1: 2, 2: 5, 3: 11}
    hour = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for t in range(seq_len):
        tk = tokens[:, t]
        for tok_id, dh in deltas.items():
            m = tk == tok_id
            if m.any():
                hour[m] = (hour[m] + dh) % 12
        rm = tk >= reset_offset
        if rm.any():
            hour[rm] = tk[rm] - reset_offset
        targets[:, t] = hour
    return tokens, targets


class CompactGearClockModel(nn.Module):
    """
    Uses ONLY d_g = 8 boxes (< 12!) with n_h = 6 reflections per step so it has the freedom
    to invent coupled sub-gears (e.g., 3-cycle + 4-cycle or signed cycles with LCM = 12)!
    """

    def __init__(self, use_wreath: bool = True, vocab_size: int = 36, d_g: int = 8, d_v: int = 24, n_h: int = 6):
        super().__init__()
        self.use_wreath = use_wreath
        self.d_g = d_g
        self.d_v = d_v
        self.n_h = n_h

        self.emb = nn.Embedding(vocab_size, 64)
        if use_wreath:
            self.to_hh_k = nn.Linear(64, n_h * d_g)
            self.to_hh_b = nn.Linear(64, n_h)

        # Value-Keeper: on a Clock Reset, erases ALL 8 gear boxes (global reset gate) & writes fresh hour marker!
        self.to_k = nn.Linear(64, d_g)
        self.to_v = nn.Linear(64, d_v)
        self.to_reset_gate = nn.Linear(64, 1)

        self.probes = nn.Parameter(torch.eye(d_g))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_g * d_v),
            nn.Linear(d_g * d_v, 64),
            nn.SiLU(),
            nn.Linear(64, 12),
        )

    def compute_token_G(self, x: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        B, L, _ = x.shape
        raw_k = self.to_hh_k(x).view(B, L, self.n_h, self.d_g)
        raw_b = self.to_hh_b(x)
        if self.training and noise_std > 0.0:
            raw_k = raw_k + torch.randn_like(raw_k) * noise_std
            raw_b = raw_b + torch.randn_like(raw_b) * noise_std

        hk = F.normalize(raw_k, dim=-1)
        hb = 2.0 * torch.sigmoid(raw_b)
        I = torch.eye(self.d_g, device=x.device).view(1, 1, self.d_g, self.d_g)
        G = I.expand(B, L, self.d_g, self.d_g)
        for j in range(self.n_h):
            Hj = I - hb[:, :, j : j + 1].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, j], hk[:, :, j])
            G = torch.matmul(Hj, G)
        return G

    def forward(self, tokens, noise_std=0.0, deadzone=0.0, fp8_eval=False, chunk_size=16, final_only=False):
        B, L = tokens.shape
        x = self.emb(tokens)

        if self.use_wreath:
            G = self.compute_token_G(x, noise_std=noise_std)
            ortho_valley = (((G ** 2).sum(dim=-1) - 1.0) ** 2).mean()
            pm1_valley = (self.d_g - (G ** 4).sum(dim=(-2, -1))).mean() / self.d_g
            magnet_loss = ortho_valley + pm1_valley
        else:
            G = None
            magnet_loss = torch.tensor(0.0, device=tokens.device)

        k = F.normalize(self.to_k(x), dim=-1)
        v = self.to_v(x)
        beta = apply_deadzone_gate(self.to_reset_gate(x), zone=deadzone, active=(deadzone > 0.0))
        probes_norm = F.normalize(self.probes, dim=-1)

        Uc = torch.eye(self.d_g, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        S = torch.zeros(B, self.d_v, self.d_g, device=x.device)
        outs = []

        for t in range(L):
            if self.use_wreath:
                Gt = quantize_fp8_e4m3(G[:, t]) if fp8_eval else G[:, t]
                Uc = fp8_matmul(Gt, Uc) if fp8_eval else torch.bmm(Gt, Uc)
                if fp8_eval and ((t + 1) % chunk_size == 0):
                    Uc = snap_to_exact_signed_perm(Uc)
                kt_eff = torch.einsum("bij,bi->bj", Uc, k[:, t])
            else:
                kt_eff = k[:, t]

            bt = beta[:, t]  # [B, 1] -> global clock reset gate in [0, 1]!
            new_write = torch.einsum("bv,bg->bvg", v[:, t], kt_eff)
            S = (1.0 - bt.unsqueeze(-1)) * S + bt.unsqueeze(-1) * new_write
            if fp8_eval and bt.max() > 0:
                S = quantize_fp8_e4m3(S)

            if not final_only or (t == L - 1):
                if self.use_wreath:
                    pt_eff = torch.einsum("bij,ri->brj", Uc, probes_norm)
                else:
                    pt_eff = probes_norm.unsqueeze(0).expand(B, -1, -1)
                read = torch.einsum("bvg,brg->brv", S, pt_eff).reshape(B, -1)
                outs.append(read)

        logits = self.classifier(torch.stack(outs, dim=1))
        return logits, magnet_loss


def inspect_learned_gears(model: CompactGearClockModel):
    """Inspects the exact 8x8 {-1, 0, +1} matrix the AI invented for '+1 hour' (Token 0)!"""
    with torch.no_grad():
        tok0 = torch.tensor([[0]])
        G_raw = model.compute_token_G(model.emb(tok0))[0, 0]
        G_snap = snap_to_exact_signed_perm(G_raw.unsqueeze(0))[0]

        # Find exact cycle period k >= 1 where G_snap^k == Identity!
        eye = torch.eye(8)
        P = eye.clone()
        period = None
        for k in range(1, 61):
            P = G_snap @ P
            if torch.equal(P, eye):
                period = k
                break

        # Trace individual coordinate orbits (the gears!)
        visited = set()
        gears = []
        for start in range(8):
            if start in visited:
                continue
            orbit = []
            curr = start
            sign_prod = 1
            while curr not in visited:
                visited.add(curr)
                orbit.append(curr)
                # Find where basis vector e_curr goes: column `curr` of G_snap
                col = G_snap[:, curr]
                nxt = int(col.abs().argmax().item())
                sgn = int(col[nxt].item())
                sign_prod *= sgn
                curr = nxt
            effective_len = len(orbit) * (2 if sign_prod == -1 else 1)
            gears.append(f"Boxes {orbit} -> {len(orbit)}-box loop (sign={sign_prod:+d} => {effective_len}-state gear!)")

        print("\n--- INSIDE THE AI'S BRAIN: Inspecting Learned '+1 Hour' 8x8 Matrix ---")
        print(f"  Exact Mathematical Repeat Period of G_('+1 hr'): {period} clicks!")
        for g_desc in gears:
            print(f"    * {g_desc}")


def run_part_5b_clock_gears():
    print("\n" + "=" * 88)
    print("PART 5B: 12-Hour Clock Inside ONLY 8 Boxes (Tested in 8-Bit FP8 @ C=4 to L=16,384!)")
    print("=" * 88)

    eval_lengths = [48, 512, 2048, 16384]
    torch.manual_seed(888)
    eval_sets = {L: generate_12hr_clock_puzzle(32, L) for L in eval_lengths}

    trained_wreath = None
    for label, use_w in [
        ("1. Standard DeltaNet (No Swapper, 8 boxes)  ", False),
        ("2. Wreath-DeltaNet (Ours, ONLY 8 boxes, FP8)", True),
    ]:
        torch.manual_seed(7)
        m = CompactGearClockModel(use_wreath=use_w, d_g=8, d_v=24, n_h=6)
        opt = torch.optim.AdamW(m.parameters(), lr=3.5e-3)
        recent_acc = 1.0 / 12.0
        t0 = time.time()

        for step in range(1, 1501):
            lam, ns = compute_adaptive_control(step, recent_acc)
            dz = 0.05 if recent_acc >= 0.75 else 0.0
            toks, tgts = generate_12hr_clock_puzzle(64, 48)
            logits, mloss = m(toks, noise_std=ns, deadzone=dz)
            loss = F.cross_entropy(logits.reshape(-1, 12), tgts.reshape(-1)) + lam * mloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                b_acc = (logits[:, -1].argmax(dim=-1) == tgts[:, -1]).float().mean().item()
                recent_acc = 0.90 * recent_acc + 0.10 * b_acc

        dt = time.time() - t0
        m.eval()
        scores = []
        with torch.no_grad():
            for L in eval_lengths:
                etoks, etgts = eval_sets[L]
                elog, _ = m(etoks, deadzone=0.05, fp8_eval=True, chunk_size=4, final_only=True)
                acc = (elog[:, 0].argmax(dim=-1) == etgts[:, -1]).float().mean().item()
                scores.append(f"L={L}:{acc*100:5.1f}%")
        print(f"  {label} ({dt:.1f}s) -> FP8 Eval: " + " | ".join(scores))
        if use_w:
            trained_wreath = m

    inspect_learned_gears(trained_wreath)


if __name__ == "__main__":
    run_part_5a_fp8_chunk_sweep()
    run_part_5b_clock_gears()
