"""
Stage 2: Self-Inventing NCM-TRM Dual-Gear Wreath Language Model (with SuperBPE + CoBPE)
=======================================================================================
Unlike Stage 1 (which supervised explicit box/clock slots), Stage 2 is a pure
Next-Concept / Autoregressive Reasoning Language Model:
  - ZERO pre-labeled boxes or clock targets! Trained ONLY on next-(base, modifier)
    token prediction over `<think> ... </think> <ans> ... </ans>` reasoning traces.
  - Uses SuperBPE + CoBPE compositional embeddings (`E_base + E_mod`) and factorized
    output heads (`P(base_t+1) * P(mod_t+1 | base_t+1)`).
  - Inside, the model autonomously invents how to use its 6x6 Swap-Gears, 2D Clock-Dials,
    Gated DeltaNet associative memory, NCM coarse chunk concepts, and 3-pass unfrozen
    TRM latent scratchpad `z_k`!
"""

import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp1_adaptive_locking import snap_to_exact_signed_perm
from exp2_deadzone_gate import apply_deadzone_gate
from exp5_fp8_and_clock_gears import fp8_matmul, quantize_fp8_e4m3
from exp6_trm_ncm_wreath import snap_2d_clock_dial_30deg

from stage2.reasoning_corpus import build_reasoning_splits
from stage2.tokenizer_super_cobpe import SuperCoBPETokenizer


class SelfInventingWreathLM(nn.Module):
    """
    Pure Language Model comparing:
      - `use_wreath_trm=False`: Matched Standard Gated-DeltaNet LM Baseline
      - `use_wreath_trm=True` : Autonomous NCM + 3-Pass TRM + Dual-Gear Wreath LM
    """

    def __init__(
        self,
        base_vocab: int,
        mod_vocab: int,
        d_model: int = 80,
        d_swap: int = 6,
        d_v: int = 32,
        n_dials: int = 4,
        chunk_c: int = 4,
        use_wreath_trm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_swap = d_swap
        self.d_v = d_v
        self.n_dials = n_dials
        self.chunk_c = chunk_c
        self.use_wreath_trm = use_wreath_trm

        # CoBPE Compositional Input Embeddings: h_0 = E_base(base) + E_mod(modifier)!
        self.base_emb = nn.Embedding(base_vocab, d_model)
        self.mod_emb = nn.Embedding(mod_vocab, d_model)
        # Causal Local Depthwise Conv1d (standard in Mamba/Gated-DeltaNet) so adjacent tokens
        # like `set a=` + ` 4` or `[2,` + `7,` + `3]` bind `(variable_key, value)` locally!
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.conv_mix = nn.Linear(d_model, d_model)

        if use_wreath_trm:
            # Stage 1 (NCM): Causal chunk-concept planner over C=4 supertoken blocks
            self.ncm_pool = nn.Sequential(
                nn.LayerNorm(chunk_c * d_model),
                nn.Linear(chunk_c * d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            self.pass_emb = nn.Embedding(4, d_model)
            # Stage 2 (TRM): Unfrozen latent scratchpad updater z_k = z_{k-1} + f(h, z_{k-1}, locked_y_{k-1})
            y_dim = d_v + 2 * n_dials
            self.trm_z_step = nn.Sequential(
                nn.LayerNorm(2 * d_model + y_dim),
                nn.Linear(2 * d_model + y_dim, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model),
            )
            self.y_conf_head = nn.Linear(y_dim, 1)  # Autonomous confidence estimator for y_k anchor lock!

            # Dual-Gear Worker 1 (receives modifier-enriched token + scratchpad z_k)
            self.w1_fusion = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.SiLU(),
            )
            self.to_hh_k = nn.Linear(d_model, 2 * d_swap)
            self.to_hh_b = nn.Linear(d_model, 2)
            self.to_clk_theta = nn.Linear(d_model, n_dials)
            with torch.no_grad():
                self.to_clk_theta.weight.mul_(0.08)
                self.to_clk_theta.bias.zero_()

        # Clean Worker 2: Associative Fast-Weight Memory + Autonomous 2D Clock-Dials
        self.w2_norm = nn.LayerNorm(d_model)
        self.to_q = nn.Linear(d_model, d_swap)
        self.to_k = nn.Linear(d_model, d_swap)
        self.to_v = nn.Linear(d_model, d_v)
        self.to_beta = nn.Linear(d_model, 1)

        self.to_clk_phi = nn.Linear(d_model, n_dials)
        self.to_clk_beta = nn.Linear(d_model, n_dials)

        mem_out_dim = d_v + 2 * n_dials
        self.mem_proj = nn.Linear(mem_out_dim, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.final_norm = nn.LayerNorm(d_model)

        # CoBPE Factorized Output Heads: Base Token Head + Base-Conditioned Modifier Head!
        self.base_head = nn.Linear(d_model, base_vocab)
        self.mod_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, mod_vocab),
        )

    def _run_memory_pass(
        self,
        h_clean: torch.Tensor,
        h_op: torch.Tensor,
        z_k: torch.Tensor | None,
        active_gears: bool,
        noise_std: float = 0.0,
        deadzone: float = 0.0,
        fp8_eval: bool = False,
    ):
        B, L, _ = h_clean.shape
        w2 = self.w2_norm(h_clean)

        q = F.normalize(self.to_q(w2), dim=-1)
        k = F.normalize(self.to_k(w2), dim=-1)
        v = self.to_v(w2)
        b_beta = apply_deadzone_gate(self.to_beta(w2), zone=deadzone, active=(deadzone > 0.0))

        phi = math.pi * torch.tanh(self.to_clk_phi(w2))
        reset_hand = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)  # [B, L, D, 2]
        c_beta = apply_deadzone_gate(self.to_clk_beta(w2), zone=deadzone, active=(deadzone > 0.0))

        if active_gears and self.use_wreath_trm:
            w1 = self.w1_fusion(torch.cat([h_op, z_k], dim=-1))
            raw_k = self.to_hh_k(w1).view(B, L, 2, self.d_swap)
            raw_b = self.to_hh_b(w1)
            raw_th = self.to_clk_theta(w1)
            if self.training and noise_std > 0.0:
                raw_k = raw_k + torch.randn_like(raw_k) * noise_std
                raw_b = raw_b + torch.randn_like(raw_b) * noise_std
                raw_th = raw_th + torch.randn_like(raw_th) * (noise_std * 0.5)

            hk = F.normalize(raw_k, dim=-1)
            hb = 2.0 * torch.sigmoid(raw_b)
            I6 = torch.eye(self.d_swap, device=h_clean.device).view(1, 1, self.d_swap, self.d_swap)
            H1 = I6 - hb[:, :, 0:1].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 0], hk[:, :, 0])
            H2 = I6 - hb[:, :, 1:2].unsqueeze(-1) * torch.einsum("bti,btj->btij", hk[:, :, 1], hk[:, :, 1])
            G_swap = torch.matmul(H2, H1)

            ortho_v = (((G_swap ** 2).sum(dim=-1) - 1.0) ** 2).mean()
            pm1_v = (self.d_swap - (G_swap ** 4).sum(dim=(-2, -1))).mean() / self.d_swap

            theta = math.pi * torch.tanh(raw_th)
            self.last_G_swap = G_swap.detach()
            self.last_theta = theta.detach()
            clk_v = (0.5 * (1.0 - torch.cos(12.0 * theta))).mean()
            mag_loss = ortho_v + pm1_v + clk_v

            ct, st = torch.cos(theta), torch.sin(theta)
            R_clk = torch.stack([torch.stack([ct, -st], dim=-1), torch.stack([st, ct], dim=-1)], dim=-2)
        else:
            mag_loss = torch.tensor(0.0, device=h_clean.device)
            G_swap = None
            R_clk = None

        U_swap = torch.eye(self.d_swap, device=h_clean.device).unsqueeze(0).repeat(B, 1, 1)
        U_clk = torch.eye(2, device=h_clean.device).view(1, 1, 2, 2).repeat(B, self.n_dials, 1, 1)
        S_box = torch.zeros(B, self.d_v, self.d_swap, device=h_clean.device)
        S_clk = torch.zeros(B, self.n_dials, 2, device=h_clean.device)

        y_outs = []
        for t in range(L):
            if active_gears and self.use_wreath_trm:
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

                kt_eff = torch.bmm(U_swap.transpose(1, 2), k[:, t].unsqueeze(-1)).squeeze(-1)
                qt_eff = torch.bmm(U_swap.transpose(1, 2), q[:, t].unsqueeze(-1)).squeeze(-1)
            else:
                kt_eff = k[:, t]
                qt_eff = q[:, t]

            bb = b_beta[:, t]
            old_b = torch.bmm(S_box, kt_eff.unsqueeze(-1)).squeeze(-1)
            S_box = S_box + ((v[:, t] - old_b) * bb).unsqueeze(-1) * kt_eff.unsqueeze(1)
            if fp8_eval and bb.max() > 0:
                S_box = quantize_fp8_e4m3(S_box)

            cb = c_beta[:, t].unsqueeze(-1)
            w_hand = torch.matmul(U_clk.transpose(-1, -2), reset_hand[:, t].unsqueeze(-1)).squeeze(-1)
            S_clk = (1.0 - cb) * S_clk + cb * w_hand
            if fp8_eval and cb.max() > 0:
                S_clk = quantize_fp8_e4m3(S_clk)

            read_box = torch.bmm(S_box, qt_eff.unsqueeze(-1)).squeeze(-1)  # [B, d_v]
            read_clk = torch.matmul(U_clk, S_clk.unsqueeze(-1)).squeeze(-1).reshape(B, -1)  # [B, 2*D]
            y_outs.append(torch.cat([read_box, read_clk], dim=-1))

        return torch.stack(y_outs, dim=1), mag_loss  # [B, L, d_v + 2*D]

    def forward(
        self,
        base_ids: torch.Tensor,
        mod_ids: torch.Tensor,
        target_base_ids: torch.Tensor | None = None,
        noise_std: float = 0.0,
        deadzone: float = 0.0,
        fp8_eval: bool = False,
    ):
        B, L = base_ids.shape
        e_base = self.base_emb(base_ids)
        e_mod = self.mod_emb(mod_ids)
        raw_tok = e_base + e_mod
        # Strictly causal Conv1d (slice `[:, :, :L]` removes right-side padding -> zero future leak!)
        conv_feat = F.silu(self.local_conv(raw_tok.transpose(1, 2))[:, :, :L].transpose(1, 2))
        h_clean = raw_tok + self.conv_mix(conv_feat)
        h_op = h_clean + 0.5 * e_mod

        if not self.use_wreath_trm:
            # Standard 1-Pass Gated DeltaNet LM Baseline
            y_seq, mag_loss = self._run_memory_pass(
                h_clean, h_op, None, active_gears=False, deadzone=deadzone, fp8_eval=fp8_eval
            )
            h_out = h_clean + self.mem_proj(y_seq)
            h_out = h_out + self.ffn(h_out)
        else:
            # Pad to chunk_c multiple for NCM coarse chunk pooling
            rem = L % self.chunk_c
            pad = (self.chunk_c - rem) % self.chunk_c
            if pad > 0:
                h_pad = F.pad(h_clean, (0, 0, 0, pad))
            else:
                h_pad = h_clean
            n_c = h_pad.shape[1] // self.chunk_c
            coarse = self.ncm_pool(h_pad.view(B, n_c, self.chunk_c * self.d_model))
            zpad_c = torch.zeros(B, 1, self.d_model, device=base_ids.device)
            causal_c = torch.cat([zpad_c, coarse[:, :-1]], dim=1)
            z = causal_c.unsqueeze(2).expand(B, n_c, self.chunk_c, self.d_model).reshape(B, -1, self.d_model)[:, :L]

            # Stage 1: NCM Coarse Skeleton Pass
            y_k, _ = self._run_memory_pass(
                h_clean, h_op, z, active_gears=False, deadzone=deadzone, fp8_eval=fp8_eval
            )
            mags = []
            # Stage 2: 2 Autonomous TRM Recursive Passes (z unfrozen, y >= 75% confidence-anchored!)
            for k_pass in range(1, 3):
                conf = torch.sigmoid(self.y_conf_head(y_k))  # [B, L, 1] learned certainty score!
                tgt_scale = torch.where(conf >= 0.75, torch.ones_like(conf), conf)
                scale = tgt_scale - conf.detach() + conf
                locked_y = y_k * scale  # NOTE: Causal! y_k[:, t] ONLY used tokens <= t!

                trm_in = torch.cat([h_op + self.pass_emb.weight[k_pass].view(1, 1, -1), z, locked_y], dim=-1)
                z = z + self.trm_z_step(trm_in)  # Scratchpad z stays 100% UNFROZEN!

                y_k, m_l = self._run_memory_pass(
                    h_clean,
                    h_op,
                    z,
                    active_gears=True,
                    noise_std=noise_std,
                    deadzone=deadzone,
                    fp8_eval=fp8_eval,
                )
                mags.append(m_l)

            mag_loss = sum(mags) / len(mags)
            h_out = h_clean + z + self.mem_proj(y_k)
            h_out = h_out + self.ffn(h_out)

        h_final = self.final_norm(h_out)
        base_logits = self.base_head(h_final)  # [B, L, base_vocab]

        # CoBPE Factorized Modifier Prediction (conditioned on next base embedding during teacher-forcing,
        # or predicted greedy base embedding at inference!)
        if target_base_ids is not None:
            next_base_e = self.base_emb(target_base_ids)
        else:
            next_base_e = self.base_emb(base_logits.argmax(dim=-1))
        mod_logits = self.mod_head(torch.cat([h_final, next_base_e], dim=-1))
        return base_logits, mod_logits, mag_loss
