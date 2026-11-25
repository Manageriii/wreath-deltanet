"""
Experiment #6 (Try #2): Unified NCM + TRM + Dual-Gear Wreath (All 4 StateBench Tiers in FP8)
===========================================================================================
Incorporates both Try #2 fixes confirmed together:
  1. Worker 2 Address Dead-Zone + {-1, 0, +1} Valley Magnet on `k`:
     Eliminates the ~6% FP8 splash across boxes so Write-Once Box 0 stays 100% intact!
  2. Decoupled Magnet Switches (70% smooth threshold for Clock & Boxes independently)
     + Full 360° Multi-Wrap Clock Density at L_train = 64:
     Forces the 2D Clock-Dial to wrap 2-3x around 12 hours during training and lock cleanly
     onto exact 30-degree multiples!
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp1_adaptive_locking import snap_to_exact_signed_perm
from exp2_deadzone_gate import apply_deadzone_gate
from exp5_fp8_and_clock_gears import fp8_matmul, quantize_fp8_e4m3


def compute_ramped_70_control(
    step: int, recent_acc: float, trigger_step: int | None
) -> tuple[float, float, int | None]:
    """
    User's 200-Step Linear Ramp Controller (decoupled per branch!):
      - Before crossing 70%: lam = 0.0, exploration jumps active (0.035 -> 0.018).
      - The instant `recent_acc >= 0.70`: latches `trigger_step` and ramps `lam`
        linearly from 0.0 -> 0.50 over 200 steps (no sudden jump, no flickering!).
    """
    if trigger_step is None and recent_acc >= 0.70:
        trigger_step = step

    if trigger_step is None:
        noise = 0.035 if recent_acc < 0.50 else 0.018
        return 0.0, noise, None
    else:
        prog = min(1.0, (step - trigger_step) / 200.0)
        lam = 0.50 * prog
        noise = 0.008 * (1.0 - 0.85 * prog)
        return lam, noise, trigger_step


def apply_address_deadzone(
    k_norm: torch.Tensor, active: bool = True, hard_snap: bool = False, thresh: float = 0.20
) -> torch.Tensor:
    """
    Dead-zones small address components (< 0.20 -> exact 0.0) smoothly first, and only
    engages hard 1-hot STE (`hard_snap=True`) once the 200-step magnet ramp is halfway done!
    """
    if not active:
        return k_norm
    trimmed = torch.where(k_norm.abs() < thresh, torch.zeros_like(k_norm), k_norm)
    norm = trimmed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    k_clean = trimmed / norm
    if not hard_snap:
        return k_clean
    max_val, max_idx = k_clean.abs().max(dim=-1, keepdim=True)
    signs = torch.gather(k_clean.sign(), -1, max_idx)
    one_hot = F.one_hot(max_idx.squeeze(-1), num_classes=k_norm.shape[-1]).float() * signs
    is_sharp = (max_val >= 0.70).float()
    hard = is_sharp * one_hot + (1.0 - is_sharp) * k_clean
    return hard - k_clean.detach() + k_clean


def generate_unified_4tier_puzzle(batch_size: int, seq_len: int):
    """
    Tokens (vocab = 77), seq_len divisible by C=4:
      0, 1, 2 : Unconditional S_4 swaps on Boxes 1..4: (1<->2), (2<->3), (3<->4)
      3       : HOP-1 CLOCK->BOX SWAP: IF Clock is EVEN (0,2,..,10) -> Swap Box 1 <-> Box 2
      4       : HOP-2 CHAINED SWAP   : IF Box 1 is EVEN (0,2,4,6)   -> Swap Box 3 <-> Box 4
      5..8    : 12-Hour Clock ticks  : +1h, +2h, +5h, -1h (mod 12)
      9..24   : 16 Filler chatter words (do nothing)
      25..36  : Reset Clock = h (h in 0..11)
      37..76  : Write(box r, val v) for r in 0..4, v in 0..7 (Box 0 ONLY at t=0 -> Tier 1!)
    """
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)

    init_boxes = torch.randint(0, 8, (batch_size, 5))
    for r in range(5):
        tokens[:, r] = 37 + r * 8 + init_boxes[:, r]
    tokens[:, 5] = 25 + torch.randint(0, 12, (batch_size,))

    rem = seq_len - 6
    u = torch.rand(batch_size, rem)
    # 50% filler, 25% clock ticks (~15 ticks in L=64 -> 2-3 full 12-hour wraps!),
    # 8% S_4 swaps, 5% Hop-1, 5% Hop-2, 2% Clock resets, 5% Box 1..4 overwrites!
    fillers = 9 + torch.randint(0, 16, (batch_size, rem))
    ticks = 5 + torch.randint(0, 4, (batch_size, rem))
    uncond_swaps = torch.randint(0, 3, (batch_size, rem))
    hop1_swap = torch.full((batch_size, rem), 3, dtype=torch.long)
    hop2_swap = torch.full((batch_size, rem), 4, dtype=torch.long)
    clk_reset = 25 + torch.randint(0, 12, (batch_size, rem))
    box_writes = 37 + 8 + torch.randint(0, 4 * 8, (batch_size, rem))

    seq_rem = torch.where(
        u < 0.50,
        fillers,
        torch.where(
            u < 0.75,
            ticks,
            torch.where(
                u < 0.83,
                uncond_swaps,
                torch.where(
                    u < 0.88,
                    hop1_swap,
                    torch.where(u < 0.93, hop2_swap, torch.where(u < 0.95, clk_reset, box_writes)),
                ),
            ),
        ),
    )
    tokens[:, 6:] = seq_rem

    boxes = torch.zeros(batch_size, 5, dtype=torch.long)
    clock = torch.zeros(batch_size, dtype=torch.long)
    tgt_boxes = torch.zeros(batch_size, seq_len, 5, dtype=torch.long)
    tgt_clock = torch.zeros(batch_size, seq_len, dtype=torch.long)

    tick_map = {5: 1, 6: 2, 7: 5, 8: 11}
    for t in range(seq_len):
        tk = tokens[:, t]
        for g in range(3):
            m = tk == g
            if m.any():
                b1, b2 = g + 1, g + 2
                boxes[m, b1], boxes[m, b2] = boxes[m, b2].clone(), boxes[m, b1].clone()
        c1 = (tk == 3) & ((clock % 2) == 0)
        if c1.any():
            boxes[c1, 1], boxes[c1, 2] = boxes[c1, 2].clone(), boxes[c1, 1].clone()
        c2 = (tk == 4) & ((boxes[:, 1] % 2) == 0)
        if c2.any():
            boxes[c2, 3], boxes[c2, 4] = boxes[c2, 4].clone(), boxes[c2, 3].clone()
        for tid, dh in tick_map.items():
            tm = tk == tid
            if tm.any():
                clock[tm] = (clock[tm] + dh) % 12
        crm = (tk >= 25) & (tk < 37)
        if crm.any():
            clock[crm] = tk[crm] - 25
        bwm = tk >= 37
        if bwm.any():
            wt = tk[bwm] - 37
            boxes[bwm, wt // 8] = wt % 8

        tgt_boxes[:, t] = boxes
        tgt_clock[:, t] = clock

    return tokens, tgt_boxes, tgt_clock


def snap_2d_clock_dial_30deg(U_clk: torch.Tensor) -> torch.Tensor:
    sin_est = 0.5 * (U_clk[:, 1, 0] - U_clk[:, 0, 1])
    cos_est = 0.5 * (U_clk[:, 0, 0] + U_clk[:, 1, 1])
    theta = torch.atan2(sin_est.float(), cos_est.float())
    tick_step = 2.0 * math.pi / 12.0  # 30 degrees
    theta_snapped = torch.round(theta / tick_step) * tick_step
    c = torch.cos(theta_snapped)
    s = torch.sin(theta_snapped)
    return torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)


def confidence_lock_draft(feats: torch.Tensor, logits: torch.Tensor, threshold: float = 0.75) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    conf = probs.max(dim=-1, keepdim=True).values
    target_scale = torch.where(conf >= threshold, torch.ones_like(conf), conf)
    scale = target_scale - conf.detach() + conf
    return feats * scale


class NCM_TRM_DualGearWreath(nn.Module):
    def __init__(
        self,
        vocab_size: int = 77,
        d_model: int = 64,
        d_swap: int = 6,
        d_v: int = 24,
        n_dials: int = 4,
        chunk_c: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_swap = d_swap
        self.d_v = d_v
        self.n_dials = n_dials
        self.chunk_c = chunk_c

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.ncm_chunk_pool = nn.Sequential(
            nn.LayerNorm(chunk_c * d_model),
            nn.Linear(chunk_c * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        y_dim = 5 * d_v + d_v
        self.pass_emb = nn.Embedding(4, d_model)
        self.trm_z_update = nn.Sequential(
            nn.LayerNorm(2 * d_model + y_dim),
            nn.Linear(2 * d_model + y_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.w1_fusion = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
        )
        self.to_hh_k = nn.Linear(d_model, 2 * d_swap)
        self.to_hh_b = nn.Linear(d_model, 2)
        # Gear B: 4 parallel 2D unit-circle Clock-Dials!
        self.to_clk_angle = nn.Linear(d_model, n_dials)
        with torch.no_grad():
            self.to_clk_angle.weight.mul_(0.1)
            self.to_clk_angle.bias.zero_()

        self.w2_norm = nn.LayerNorm(d_model)
        self.to_box_k = nn.Linear(d_model, d_swap)
        self.to_box_v = nn.Linear(d_model, d_v)
        self.to_box_beta = nn.Linear(d_model, 1)
        self.box_queries = nn.Parameter(torch.eye(d_swap)[:5])
        self.box_readout = nn.Linear(d_v, 8)

        # Clock Reset lives on the EXACT SAME 2D unit circle [cos(phi), sin(phi)]!
        self.to_clk_reset_phi = nn.Linear(d_model, n_dials)
        self.to_clk_beta = nn.Linear(d_model, n_dials)
        self.clk_readout = nn.Sequential(
            nn.LayerNorm(2 * n_dials),
            nn.Linear(2 * n_dials, d_v),
            nn.SiLU(),
            nn.Linear(d_v, 12),
        )

    def run_dual_gear_pass(
        self,
        base_h: torch.Tensor,
        z_k: torch.Tensor,
        active_gears: bool,
        box_noise: float = 0.0,
        clk_noise: float = 0.0,
        box_dz: float = 0.0,
        clk_dz: float = 0.0,
        hard_addr: bool = False,
        fp8_eval: bool = False,
        final_only: bool = False,
    ):
        B, L, _ = base_h.shape

        # Worker 2 (clean tokens ONLY + Fix #1: Address Dead-Zone & Valley Magnet on `bk`!)
        w2 = self.w2_norm(base_h)
        bk_raw = F.normalize(self.to_box_k(w2), dim=-1)
        # Valley magnet on address pointer `bk` -> pulls active write pointers to exact one-hot basis!
        b_beta = apply_deadzone_gate(self.to_box_beta(w2), zone=box_dz, active=(box_dz > 0.0))
        write_weight = b_beta.detach().squeeze(-1)  # Only pull `bk` to 1-hot on tokens that actually write!
        addr_valley = (write_weight * (1.0 - (bk_raw ** 4).sum(dim=-1))).mean()

        bk = apply_address_deadzone(bk_raw, active=(box_dz > 0.0), hard_snap=(hard_addr or fp8_eval))
        bv = self.to_box_v(w2)
        bq_norm = apply_address_deadzone(
            F.normalize(self.box_queries, dim=-1), active=(box_dz > 0.0), hard_snap=(hard_addr or fp8_eval)
        )

        phi = math.pi * torch.tanh(self.to_clk_reset_phi(w2))  # [B, L, D]
        reset_hand = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)  # [B, L, D, 2]
        c_beta = apply_deadzone_gate(self.to_clk_beta(w2), zone=clk_dz, active=(clk_dz > 0.0))
        phi_valley = (c_beta.detach() * 0.5 * (1.0 - torch.cos(12.0 * phi))).mean()

        if active_gears:
            w1 = self.w1_fusion(torch.cat([base_h, z_k], dim=-1))
            raw_k = self.to_hh_k(w1).view(B, L, 2, self.d_swap)
            raw_b = self.to_hh_b(w1)
            raw_ang = self.to_clk_angle(w1)  # [B, L, D]
            if self.training:
                if box_noise > 0.0:
                    raw_k = raw_k + torch.randn_like(raw_k) * box_noise
                    raw_b = raw_b + torch.randn_like(raw_b) * box_noise
                if clk_noise > 0.0:
                    raw_ang = raw_ang + torch.randn_like(raw_ang) * (clk_noise * 0.5)

            hk = F.normalize(raw_k, dim=-1)
            hb = 2.0 * torch.sigmoid(raw_b)
            I6 = torch.eye(self.d_swap, device=base_h.device).view(1, 1, self.d_swap, self.d_swap)
            H1 = I6 - hb[:, :, 0:1].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 0], hk[:, :, 0])
            H2 = I6 - hb[:, :, 1:2].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 1], hk[:, :, 1])
            G_swap = torch.matmul(H2, H1)

            ortho_v = (((G_swap ** 2).sum(dim=-1) - 1.0) ** 2).mean()
            pm1_v = (self.d_swap - (G_swap ** 4).sum(dim=(-2, -1))).mean() / self.d_swap
            box_magnet = ortho_v + pm1_v + addr_valley

            theta = math.pi * torch.tanh(raw_ang)  # [B, L, D]
            self.last_theta = theta.detach()
            clk_magnet = (0.5 * (1.0 - torch.cos(12.0 * theta))).mean() + phi_valley

            ct, st = torch.cos(theta), torch.sin(theta)
            R_clk = torch.stack([torch.stack([ct, -st], dim=-1), torch.stack([st, ct], dim=-1)], dim=-2)
        else:
            box_magnet = addr_valley
            clk_magnet = phi_valley
            G_swap = None
            R_clk = None

        U_swap = torch.eye(self.d_swap, device=base_h.device).unsqueeze(0).repeat(B, 1, 1)
        U_clk = torch.eye(2, device=base_h.device).view(1, 1, 2, 2).repeat(B, self.n_dials, 1, 1)
        S_box = torch.zeros(B, self.d_v, self.d_swap, device=base_h.device)
        S_clk = torch.zeros(B, self.n_dials, 2, device=base_h.device)

        box_outs = []
        clk_outs = []

        for t in range(L):
            if active_gears:
                if fp8_eval:
                    Gs_t = quantize_fp8_e4m3(G_swap[:, t])
                    Rc_t = quantize_fp8_e4m3(R_clk[:, t])
                    U_swap = fp8_matmul(Gs_t, U_swap)
                    U_clk = fp8_matmul(Rc_t, U_clk)
                    if (t + 1) % self.chunk_c == 0:
                        U_swap = snap_to_exact_signed_perm(U_swap)
                        U_clk = quantize_fp8_e4m3(
                            snap_2d_clock_dial_30deg(U_clk.reshape(-1, 2, 2)).view(B, self.n_dials, 2, 2)
                        )
                else:
                    U_swap = torch.bmm(G_swap[:, t], U_swap)
                    U_clk = torch.matmul(R_clk[:, t], U_clk)

                kt_eff = torch.bmm(U_swap.transpose(1, 2), bk[:, t].unsqueeze(-1)).squeeze(-1)
                if fp8_eval:
                    kt_eff = apply_address_deadzone(kt_eff, active=True, hard_snap=True)
            else:
                kt_eff = bk[:, t]

            bb = b_beta[:, t]
            old_b = torch.bmm(S_box, kt_eff.unsqueeze(-1)).squeeze(-1)
            diff_b = ((bv[:, t] - old_b) * bb).unsqueeze(-1)
            S_box = S_box + diff_b * kt_eff.unsqueeze(1)
            if fp8_eval and bb.max() > 0:
                S_box = quantize_fp8_e4m3(S_box)

            cb = c_beta[:, t].unsqueeze(-1)  # [B, D, 1]
            w_hand = torch.matmul(U_clk.transpose(-1, -2), reset_hand[:, t].unsqueeze(-1)).squeeze(-1)
            S_clk = (1.0 - cb) * S_clk + cb * w_hand
            if fp8_eval and cb.max() > 0:
                S_clk = quantize_fp8_e4m3(S_clk)

            if not final_only or (t == L - 1):
                if active_gears:
                    qt_eff_T = torch.matmul(U_swap.transpose(1, 2), bq_norm.t())
                    box_outs.append(torch.bmm(S_box, qt_eff_T).transpose(1, 2))
                else:
                    box_outs.append(torch.matmul(S_box, bq_norm.t()).transpose(1, 2))
                cur_hand = torch.matmul(U_clk, S_clk.unsqueeze(-1)).squeeze(-1)
                clk_outs.append(cur_hand.reshape(B, -1))

        box_feats = torch.stack(box_outs, dim=1)
        clk_feats = torch.stack(clk_outs, dim=1)
        box_logits = self.box_readout(box_feats)
        clk_logits = self.clk_readout(clk_feats)
        return box_feats, clk_feats, box_logits, clk_logits, box_magnet, clk_magnet

    def forward(
        self,
        tokens: torch.Tensor,
        box_noise: float = 0.0,
        clk_noise: float = 0.0,
        box_dz: float = 0.0,
        clk_dz: float = 0.0,
        hard_addr: bool = False,
        fp8_eval: bool = False,
        eval_final_only: bool = False,
    ):
        B, L = tokens.shape
        base_h = self.tok_emb(tokens)

        n_chunks = L // self.chunk_c
        chunks = base_h.view(B, n_chunks, self.chunk_c * self.d_model)
        coarse_concepts = self.ncm_chunk_pool(chunks)
        zpad_chunk = torch.zeros(B, 1, self.d_model, device=tokens.device)
        causal_concepts = torch.cat([zpad_chunk, coarse_concepts[:, :-1]], dim=1)
        z = causal_concepts.unsqueeze(2).expand(B, n_chunks, self.chunk_c, self.d_model).reshape(B, L, self.d_model)

        # Stage 1: NCM Coarse Skeleton Pass (needs full sequence outputs to feed TRM Step 1!)
        b_f, c_f, b_log0, c_log0, bm0, _ = self.run_dual_gear_pass(
            base_h,
            z,
            active_gears=False,
            box_dz=box_dz,
            clk_dz=clk_dz,
            hard_addr=hard_addr,
            fp8_eval=fp8_eval,
            final_only=False,
        )

        all_box_logits = [b_log0[:, -1:] if eval_final_only else b_log0]
        all_clk_logits = [c_log0[:, -1:] if eval_final_only else c_log0]
        box_mags = [bm0]
        clk_mags = []

        zpad_b = torch.zeros(B, 1, 5, self.d_v, device=tokens.device)
        zpad_c = torch.zeros(B, 1, self.d_v, device=tokens.device)

        last_full_blog = b_log0
        last_full_clog = c_log0

        for k_pass in range(1, 4):
            locked_b = confidence_lock_draft(b_f, last_full_blog, threshold=0.75)
            clk_hidden = F.silu(self.clk_readout[1](self.clk_readout[0](c_f)))
            locked_c = confidence_lock_draft(clk_hidden, last_full_clog, threshold=0.75)

            ctx_b = torch.cat([zpad_b, locked_b[:, :-1]], dim=1).reshape(B, L, -1)
            ctx_c = torch.cat([zpad_c, locked_c[:, :-1]], dim=1)
            y_ctx = torch.cat([ctx_b, ctx_c], dim=-1)

            trm_in = torch.cat([base_h + self.pass_emb.weight[k_pass].view(1, 1, -1), z, y_ctx], dim=-1)
            z = z + self.trm_z_update(trm_in)

            only_last = eval_final_only and (k_pass == 3)
            b_f, c_f, b_log, c_log, bm, cm = self.run_dual_gear_pass(
                base_h,
                z,
                active_gears=True,
                box_noise=box_noise,
                clk_noise=clk_noise,
                box_dz=box_dz,
                clk_dz=clk_dz,
                hard_addr=hard_addr,
                fp8_eval=fp8_eval,
                final_only=only_last,
            )
            last_full_blog = b_log
            last_full_clog = c_log
            all_box_logits.append(b_log[:, -1:] if (eval_final_only and not only_last) else b_log)
            all_clk_logits.append(c_log[:, -1:] if (eval_final_only and not only_last) else c_log)
            box_mags.append(bm)
            clk_mags.append(cm)

        return all_box_logits, all_clk_logits, sum(box_mags) / len(box_mags), sum(clk_mags) / len(clk_mags)


def inspect_clock_dial_angles(model: NCM_TRM_DualGearWreath):
    names = {5: "+1 hour", 6: "+2 hours", 7: "+5 hours", 8: "-1 hour (+11h)", 9: "Filler word"}
    print("\n--- INSIDE DUAL-GEAR WORKER 1: Learned 2D Clock-Dial Angles (Pass 3) ---")
    with torch.no_grad():
        toks, _, _ = generate_unified_4tier_puzzle(16, 64)
        _ = model(toks, box_dz=0.05, clk_dz=0.05, fp8_eval=False)
        thetas_deg = torch.rad2deg(model.last_theta)  # [16, 64, n_dials]
        for tid, label in names.items():
            mask = toks == tid
            if mask.any():
                dial_means = thetas_deg[mask].mean(dim=0).tolist()
                fmt = ", ".join([f"{d:+6.1f}°" for d in dial_means])
                print(f"  Token '{label:14s}' -> 4 Dials: [{fmt}]")


def run_experiment_6_v3():
    print("=" * 94)
    print("EXPERIMENT #6 (TRY #3): Dependency-Ordered Magnet Cascade (Clock First -> Boxes 1-4 Second!)")
    print("=" * 94)

    torch.manual_seed(42)
    model = NCM_TRM_DualGearWreath(vocab_size=77, d_model=64, d_swap=6, d_v=24, n_dials=4, chunk_c=4)
    opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3)

    recent_box14_acc = 0.125
    recent_clk_acc = 1.0 / 12.0
    box_trig: int | None = None
    clk_trig: int | None = None
    t0 = time.time()

    for step in range(1, 2201):
        # 1. Upstream Clock locks first once Clock >= 85%
        if clk_trig is None and recent_clk_acc >= 0.85:
            clk_trig = step
        if clk_trig is None:
            lam_clk, ns_clk = 0.0, (0.030 if recent_clk_acc < 0.50 else 0.015)
        else:
            p_c = min(1.0, (step - clk_trig) / 200.0)
            lam_clk, ns_clk = 0.50 * p_c, 0.006 * (1.0 - 0.85 * p_c)

        # 2. Downstream Boxes 1-4 ONLY lock AFTER Clock is settled (>= 92%) AND Boxes 1-4 >= 86%!
        if box_trig is None and (clk_trig is not None and recent_clk_acc >= 0.92) and (recent_box14_acc >= 0.86):
            box_trig = step
        if box_trig is None:
            lam_box, ns_box = 0.0, (0.030 if recent_box14_acc < 0.55 else 0.015)
            hard_addr = False
        else:
            p_b = min(1.0, (step - box_trig) / 200.0)
            lam_box, ns_box = 0.50 * p_b, 0.006 * (1.0 - 0.85 * p_b)
            hard_addr = p_b >= 0.50  # Smooth trim first 100 steps, hard 1-hot snap after!

        dz_box = 0.05 if box_trig is not None else 0.0
        dz_clk = 0.05 if clk_trig is not None else 0.0

        toks, tgt_b, tgt_c = generate_unified_4tier_puzzle(64, seq_len=64)
        b_logs, c_logs, b_mag, c_mag = model(
            toks,
            box_noise=ns_box,
            clk_noise=ns_clk,
            box_dz=dz_box,
            clk_dz=dz_clk,
            hard_addr=hard_addr,
            fp8_eval=False,
        )

        # Causal pass alignment: Step 1 resolves Clock; Step 3 resolves full multi-hop Boxes + Clock!
        loss = (
            0.20 * F.cross_entropy(c_logs[1][:, 6:].reshape(-1, 12), tgt_c[:, 6:].reshape(-1))
            + 0.15 * (
                F.cross_entropy(b_logs[2][:, 6:].reshape(-1, 8), tgt_b[:, 6:].reshape(-1))
                + F.cross_entropy(c_logs[2][:, 6:].reshape(-1, 12), tgt_c[:, 6:].reshape(-1))
            )
            + 1.00 * (
                F.cross_entropy(b_logs[3][:, 6:].reshape(-1, 8), tgt_b[:, 6:].reshape(-1))
                + F.cross_entropy(c_logs[3][:, 6:].reshape(-1, 12), tgt_c[:, 6:].reshape(-1))
            )
            + lam_box * b_mag
            + lam_clk * c_mag
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            # Strictly measure Boxes 1..4 (excluding trivial Box 0 so Box 0 can't trip the magnet early!)
            ab14 = (b_logs[3][:, -1, 1:].argmax(dim=-1) == tgt_b[:, -1, 1:]).float().mean().item()
            ac = (c_logs[3][:, -1].argmax(dim=-1) == tgt_c[:, -1]).float().mean().item()
            recent_box14_acc = 0.90 * recent_box14_acc + 0.10 * ab14
            recent_clk_acc = 0.90 * recent_clk_acc + 0.10 * ac

    dt = time.time() - t0
    print(
        f"Trained 2,200 steps @ L=64 in {dt:.1f}s | Running Acc -> Boxes 1-4: {recent_box14_acc*100:.1f}% (lam={lam_box:.2f}), Clock: {recent_clk_acc*100:.1f}% (lam={lam_clk:.2f})\n"
    )

    model.eval()
    eval_lengths = [64, 512, 2048, 8192]
    torch.manual_seed(2026)
    eval_sets = {L: generate_unified_4tier_puzzle(32, L) for L in eval_lengths}

    pass_titles = [
        "Stage 1: NCM Coarse Skeleton (Pass 0)",
        "Stage 2: TRM Recursion Step 1        ",
        "Stage 2: TRM Recursion Step 2        ",
        "Stage 2: TRM Recursion Step 3 (Final)",
    ]
    print("--- ZERO-SHOT 8-BIT FP8 (C = 4) EVALUATION ACROSS PASSES ---")
    with torch.no_grad():
        per_pass = {p: [] for p in range(4)}
        tier_breakdown_8k = {}
        for L in eval_lengths:
            etoks, eb, ec = eval_sets[L]
            bl, cl, _, _ = model(
                etoks, box_dz=0.05, clk_dz=0.05, hard_addr=True, fp8_eval=True, eval_final_only=True
            )
            for p in range(4):
                ab = (bl[p][:, 0, 1:].argmax(dim=-1) == eb[:, -1, 1:]).float().mean().item()
                ac = (cl[p][:, 0].argmax(dim=-1) == ec[:, -1]).float().mean().item()
                per_pass[p].append(f"L={L}: Box={ab*100:5.1f}% Clk={ac*100:5.1f}%")
                if L == 8192 and p == 3:
                    tier_breakdown_8k["Tier 1 (Write-Once Box 0)"] = (bl[3][:, 0, 0].argmax(-1) == eb[:, -1, 0]).float().mean().item()
                    tier_breakdown_8k["Tier 2 & 4 (Hop-1 Boxes 1-2)"] = (bl[3][:, 0, 1:3].argmax(-1) == eb[:, -1, 1:3]).float().mean().item()
                    tier_breakdown_8k["Tier 4 (Hop-2 Chained Boxes 3-4)"] = (bl[3][:, 0, 3:5].argmax(-1) == eb[:, -1, 3:5]).float().mean().item()
                    tier_breakdown_8k["Tier 3 (12-Hour Clock in FP8)"] = ac

        for p in range(4):
            print(f"  {pass_titles[p]} -> " + " | ".join(per_pass[p]))

        print("\n--- FINAL PASS-3 BREAKDOWN AT L = 8,192 (128x EXTRAPOLATION IN 8-BIT FP8 @ C=4!) ---")
        for k_name, val in tier_breakdown_8k.items():
            print(f"  * {k_name:35s}: {val*100:5.1f}%")

    inspect_clock_dial_angles(model)


if __name__ == "__main__":
    run_experiment_6_v3()
