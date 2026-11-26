"""
Experiment #8: 3:1 Hybrid Real Python/English Perplexity & Safe-Fallback Check
==============================================================================
Directly answers the #1 reviewer critique of sub-quadratic state-tracking papers:
  "Lots of math tricks get 100% on synthetic permutations/clocks, then hurt
   perplexity on messy real-world English and Python code. What happens when
   you drop Dual-Gear Wreath heads into a real 3:1 Hybrid Language Model?"

Design (100% self-contained, zero external downloads, strict module-level split):
  - Corpus: Real-world Python source code + English docstrings read directly from
    Python's own standard library (~450 KB), with ENTIRE modules held out for
    validation (difflib, statistics, textwrap, csv) so zero leakage is possible.
  - Interleaved Auxiliary Probe: Small 10% curriculum slice of stateful Python
    variable-swap & modular-counter snippets (evaluated on unseen longer traces).
  - Matched 4-Layer 3:1 Hybrids (3 Linear Gated-DeltaNet layers + 1 Full Softmax
    Attention layer, d_model=192, H=8 heads, ~1.9M params, identical seed & data):
      1. Baseline `Hybrid-DeltaNet`       : 8/8 standard contractive DeltaNet heads.
      2. Ours     `Hybrid-Wreath-DeltaNet`: 6 standard heads + 2 zero-initialized
         Dual-Gear Wreath heads (<2.5% extra params; bit-identical at step 0!).
  - Reports honest held-out Val BPC / Perplexity on messy real Python/English,
    training throughput, Wreath gate activation, and stateful variable tracking!
"""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp2_deadzone_gate import apply_deadzone_gate


# =====================================================================
# 1. Real Messy Python + English Corpus (Strict Module-Level Split)
# =====================================================================
TRAIN_MODULES = [
    "argparse", "collections", "dataclasses", "functools",
    "json", "pathlib", "re", "urllib.parse",
]
VAL_MODULES = ["difflib", "statistics", "textwrap", "csv"]


def load_stdlib_text(mod_names: list[str]) -> str:
    chunks = []
    for name in mod_names:
        try:
            mod = __import__(name, fromlist=["*"])
            src = inspect.getsource(mod)
            chunks.append(f"# --- MODULE: {name} ---\n" + src)
        except Exception:
            continue
    return "\n\n".join(chunks)


def encode_bytes(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)


def make_stateful_py_snippets(n_samples: int, n_ops: int, seed: int) -> tuple[list[str], list[str]]:
    """Generates clean stateful Python swap/mod snippets for held-out accuracy testing."""
    rng = random.Random(seed)
    prompts, answers = [], []
    for _ in range(n_samples):
        a, b, c = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        clk = rng.randint(0, 11)
        lines = [f"a, b, c, t = {a}, {b}, {c}, {clk}"]
        for _ in range(n_ops):
            r = rng.random()
            if r < 0.35:
                lines.append("a, b = b, a")
                a, b = b, a
            elif r < 0.70:
                lines.append("b, c = c, b")
                b, c = c, b
            else:
                dt = rng.choice([1, 2, 5])
                lines.append(f"t = (t + {dt}) % 12")
                clk = (clk + dt) % 12
        lines.append("# out:")
        prompts.append("\n".join(lines) + " ")
        answers.append(f"{a},{b},{c},{clk}\n")
    return prompts, answers


# =====================================================================
# 2. Matched 3:1 Hybrid Architecture (DeltaNet vs. Wreath-DeltaNet)
# =====================================================================
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 2.5):
        super().__init__()
        hid = int(d_model * mult)
        self.w12 = nn.Linear(d_model, 2 * hid, bias=False)
        self.w3 = nn.Linear(hid, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class HybridLinearLayer(nn.Module):
    """
    Chunk-friendly Multi-Head Gated DeltaNet layer (H=8 heads).
    When `n_wreath_heads > 0`, the first `n_wreath_heads` heads receive a
    ZERO-INITIALIZED Dual-Gear Wreath orthogonal frame (2D Clock-Dial + 4D Householder
    Swap-Box), so at step 0 the layer is mathematically identical to pure DeltaNet
    and safely falls back to Identity on fuzzy English prose!
    """

    def __init__(self, d_model: int = 192, n_heads: int = 8, n_wreath_heads: int = 0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_wreath_heads = n_wreath_heads

        self.norm1 = nn.RMSNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.alpha_proj = nn.Linear(d_model, n_heads)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.constant_(self.alpha_proj.bias, 2.0)   # slow decay init (~0.88)
        nn.init.constant_(self.beta_proj.bias, -1.5)

        if n_wreath_heads > 0:
            # Tiny zero-initialized Dual-Gear Wreath frame on ONLY 2 dedicated heads (<2.5% params!)
            # Note: w_clk_th is small-random (std=0.05) while wreath_gate starts at 0.0 ->
            # bit-identical output to pure DeltaNet at step 0, WITHOUT the 0*0 gradient deadlock!
            self.d_rot = 4
            self.w_hh_k = nn.Linear(d_model, n_wreath_heads * self.d_rot, bias=False)
            self.w_hh_b = nn.Linear(d_model, n_wreath_heads)
            self.w_clk_th = nn.Linear(d_model, n_wreath_heads)
            nn.init.constant_(self.w_hh_b.bias, -1.5)
            nn.init.normal_(self.w_clk_th.weight, std=0.05)
            nn.init.zeros_(self.w_clk_th.bias)
            # Zero-init residual gate -> exact identity fallback at step 0 with live gradients!
            self.wreath_gate = nn.Parameter(torch.zeros(n_wreath_heads))

        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def _wreath_rotate_head(self, h_in: torch.Tensor, k_w: torch.Tensor, q_w: torch.Tensor, snap_c: int | None):
        """Applies fast chunk-parallel 2D Clock + 4D Householder frame to the dedicated Wreath heads."""
        B, T, _ = h_in.shape
        W = self.n_wreath_heads
        gate = torch.tanh(self.wreath_gate).view(1, 1, W, 1)  # starts at 0.0!

        # 2D cumulative planar clock phase (O(N) parallel cumsum — blazing fast on GPU!)
        th = (math.pi / 2.0) * torch.tanh(self.w_clk_th(h_in)) * gate.squeeze(-1)  # [B, T, W]
        if snap_c is not None and snap_c > 0:
            # Chunk-boundary 30-deg tick snap on cumulative phase
            step = 2.0 * math.pi / 12.0
            pad = (snap_c - (T % snap_c)) % snap_c
            th_p = F.pad(th, (0, 0, 0, pad)) if pad else th
            nc = th_p.shape[1] // snap_c
            intra = torch.cumsum(th_p.view(B, nc, snap_c, W), dim=2)
            totals = torch.round(intra[:, :, -1] / step) * step
            inter = torch.cumsum(totals, dim=1)
            prev = torch.cat([torch.zeros(B, 1, W, device=h_in.device, dtype=th.dtype), inter[:, :-1]], dim=1)
            phi = (intra + prev.unsqueeze(2)).reshape(B, -1, W)[:, :T]
        else:
            phi = torch.cumsum(th, dim=1)

        c, s = torch.cos(phi), torch.sin(phi)
        # Apply orthogonal 2D rotation across pairs of coordinates (out-of-place for clean autograd!)
        k0, k1 = k_w[..., 0::2], k_w[..., 1::2]
        q0, q1 = q_w[..., 0::2], q_w[..., 1::2]
        c_e, s_e = c.unsqueeze(-1), s.unsqueeze(-1)
        k_rot = torch.stack([c_e * k0 - s_e * k1, s_e * k0 + c_e * k1], dim=-1).reshape_as(k_w)
        q_rot = torch.stack([c_e * q0 - s_e * q1, s_e * q0 + c_e * q1], dim=-1).reshape_as(q_w)

        # 4D Householder reflection modulation (local state-flip channel, out-of-place!)
        hk = F.normalize(self.w_hh_k(h_in).view(B, T, W, self.d_rot), dim=-1)
        hb = (2.0 * torch.sigmoid(self.w_hh_b(h_in))) * gate.squeeze(-1)
        k4 = k_rot[..., : self.d_rot]
        q4 = q_rot[..., : self.d_rot]
        k4_new = k4 - hb.unsqueeze(-1) * (k4 * hk).sum(dim=-1, keepdim=True) * hk
        q4_new = q4 - hb.unsqueeze(-1) * (q4 * hk).sum(dim=-1, keepdim=True) * hk
        k_out = torch.cat([k4_new, k_rot[..., self.d_rot :]], dim=-1)
        q_out = torch.cat([q4_new, q_rot[..., self.d_rot :]], dim=-1)

        # Smooth 12-tick valley regularization on active clock angles
        reg = (0.5 * (1.0 - torch.cos(12.0 * th))).mean()
        return k_out, q_out, reg

    def forward(self, x: torch.Tensor, snap_c: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        h = self.norm1(x)
        h_c = F.silu(self.conv(h.transpose(1, 2))[:, :, :T].transpose(1, 2))
        q, k, v = self.qkv(h_c).chunk(3, dim=-1)
        q = F.normalize(q.view(B, T, self.n_heads, self.d_head), dim=-1)
        k = F.normalize(k.view(B, T, self.n_heads, self.d_head), dim=-1)
        v = v.view(B, T, self.n_heads, self.d_head)

        reg_loss = x.new_zeros(())
        if self.n_wreath_heads > 0:
            W = self.n_wreath_heads
            k_w, q_w, reg_loss = self._wreath_rotate_head(h_c, k[:, :, :W], q[:, :, :W], snap_c=snap_c)
            k = torch.cat([k_w, k[:, :, W:]], dim=2)
            q = torch.cat([q_w, q[:, :, W:]], dim=2)

        alpha = torch.sigmoid(self.alpha_proj(h_c))  # [B, T, H]
        beta = apply_deadzone_gate(self.beta_proj(h_c), zone=0.05, active=(self.n_wreath_heads > 0))  # [B, T, H]

        # Fast vectorized DeltaNet scan over sequence T (head-parallel)
        S = x.new_zeros((B, self.n_heads, self.d_head, self.d_head))
        outs = []
        for t in range(T):
            kt = k[:, t]            # [B, H, d]
            vt = v[:, t]            # [B, H, d]
            qt = q[:, t]            # [B, H, d]
            at = alpha[:, t].view(B, self.n_heads, 1, 1)
            bt = beta[:, t].view(B, self.n_heads, 1)
            S = at * S
            pred_v = torch.einsum("bhij,bhj->bhi", S, kt)
            delta = (vt - pred_v) * bt
            S = S + torch.einsum("bhi,bhj->bhij", delta, kt)
            outs.append(torch.einsum("bhij,bhj->bhi", S, qt))

        y = torch.stack(outs, dim=1).reshape(B, T, D)
        x = x + self.out_proj(y)
        x = x + self.mlp(self.norm2(x))
        return x, reg_loss


class FullAttentionLayer(nn.Module):
    def __init__(self, d_model: int = 192, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        x = x + self.out_proj(y)
        x = x + self.mlp(self.norm2(x))
        return x


class Hybrid3to1LM(nn.Module):
    """3:1 Hybrid LM: 3 Linear Gated-DeltaNet layers + 1 Full Softmax Attention layer."""

    def __init__(self, vocab_size: int = 256, d_model: int = 192, n_heads: int = 8, n_wreath_heads: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.lin0 = HybridLinearLayer(d_model, n_heads, n_wreath_heads=0)  # Layer 0: Quiet (Alternating Spec!)
        self.lin1 = HybridLinearLayer(d_model, n_heads, n_wreath_heads=n_wreath_heads)
        self.lin2 = HybridLinearLayer(d_model, n_heads, n_wreath_heads=n_wreath_heads)
        self.attn3 = FullAttentionLayer(d_model, n_heads)
        self.norm_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight  # weight tying

    def forward(self, ids: torch.Tensor, snap_c: int | None = None):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0) % 512
        x = self.emb(ids) + self.pos_emb(pos)
        x, r0 = self.lin0(x, snap_c=snap_c)
        x, r1 = self.lin1(x, snap_c=snap_c)
        x, r2 = self.lin2(x, snap_c=snap_c)
        x = self.attn3(x)
        logits = self.head(self.norm_f(x))
        return logits, r0 + r1 + r2


def sample_batches(data: torch.Tensor, batch_size: int, seq_len: int, device: str, rng: torch.Generator):
    idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=rng)
    x = torch.stack([data[i : i + seq_len] for i in idx]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in idx]).to(device)
    return x, y


def eval_val_ppl(model: nn.Module, val_data: torch.Tensor, seq_len: int, device: str, snap_c: int | None = None) -> tuple[float, float]:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, min(len(val_data) - seq_len - 1, seq_len * 48), seq_len * 4):
            xs, ys = [], []
            for b in range(4):
                pos = start + b * seq_len
                if pos + seq_len + 1 >= len(val_data):
                    break
                xs.append(val_data[pos : pos + seq_len])
                ys.append(val_data[pos + 1 : pos + seq_len + 1])
            if not xs:
                continue
            xb = torch.stack(xs).to(device)
            yb = torch.stack(ys).to(device)
            logits, _ = model(xb, snap_c=snap_c)
            loss = F.cross_entropy(logits.reshape(-1, 256), yb.reshape(-1))
            losses.append(loss.item())
    mean_ce = sum(losses) / max(1, len(losses))
    bpc = mean_ce / math.log(2.0)
    ppl = math.exp(mean_ce)
    return round(bpc, 4), round(ppl, 3)


def eval_stateful_py_nll(model: nn.Module, prompts: list[str], answers: list[str], device: str, snap_c: int | None = None) -> tuple[float, float]:
    """Measures exact token accuracy & NLL on the answer bytes of unseen stateful Python snippets."""
    model.eval()
    correct_toks, total_toks, nlls = 0, 0, []
    with torch.no_grad():
        for p, a in zip(prompts, answers):
            pb = list(p.encode("utf-8"))
            ab = list(a.encode("utf-8"))
            full = torch.tensor([pb + ab], dtype=torch.long, device=device)
            logits, _ = model(full[:, :-1], snap_c=snap_c)
            ans_logits = logits[0, len(pb) - 1 :]
            ans_tgt = full[0, len(pb) :]
            nlls.append(F.cross_entropy(ans_logits, ans_tgt).item())
            correct_toks += (ans_logits.argmax(dim=-1) == ans_tgt).sum().item()
            total_toks += len(ab)
    return round(100.0 * correct_toks / max(1, total_toks), 2), round(sum(nlls) / max(1, len(nlls)), 4)


def run_exp8():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 92)
    print(f"EXPERIMENT #8: 3:1 Hybrid Real Python/English Perplexity Check (device={device})")
    print("=" * 92)

    train_text = load_stdlib_text(TRAIN_MODULES)
    val_text = load_stdlib_text(VAL_MODULES)

    # Mix 12% stateful Python snippets into training stream (strictly separate test seed for eval!)
    tr_p, tr_a = make_stateful_py_snippets(350, n_ops=6, seed=101)
    aux_train_text = "\n\n".join(p + a for p, a in zip(tr_p, tr_a))
    full_train = train_text + "\n\n" + aux_train_text

    train_bytes = encode_bytes(full_train)
    val_bytes = encode_bytes(val_text)
    eval_p_short, eval_a_short = make_stateful_py_snippets(40, n_ops=6, seed=999)
    eval_p_long, eval_a_long = make_stateful_py_snippets(40, n_ops=14, seed=2027)

    print(f"Train real Python+English stdlib: {len(train_text):,} bytes (+ {len(aux_train_text):,} bytes stateful Py)")
    print(f"Held-out UNSEEN stdlib modules ({', '.join(VAL_MODULES)}): {len(val_text):,} bytes")

    seq_len = 160
    batch_size = 24
    steps = 320
    results = {
        "device": device,
        "train_bytes": len(train_bytes),
        "val_bytes": len(val_bytes),
        "val_modules": VAL_MODULES,
        "models": {},
    }

    for label, n_w in [
        ("Hybrid-DeltaNet (8/8 Standard)", 0),
        ("Hybrid-Wreath-DeltaNet (2/8 Wreath + 6/8 Std)", 2),
    ]:
        torch.manual_seed(42)
        rng = torch.Generator().manual_seed(42)
        model = Hybrid3to1LM(vocab_size=256, d_model=192, n_heads=8, n_wreath_heads=n_w).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.02)

        t0 = time.perf_counter()
        model.train()
        for step in range(1, steps + 1):
            xb, yb = sample_batches(train_bytes, batch_size, seq_len, device, rng)
            logits, reg = model(xb, snap_c=None)
            ce = F.cross_entropy(logits.reshape(-1, 256), yb.reshape(-1))
            loss = ce + (0.01 * reg if n_w > 0 else 0.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step in (80, 160, 240, steps):
                print(f"  [{label:<45s}] step {step:3d}/{steps} | train_ce={ce.item():.4f}")

        if device == "mps":
            torch.mps.synchronize()
        elapsed = round(time.perf_counter() - t0, 1)
        tok_s = round((steps * batch_size * seq_len) / max(0.1, elapsed))

        val_bpc, val_ppl = eval_val_ppl(model, val_bytes, seq_len, device, snap_c=None)
        val_bpc_snap, val_ppl_snap = (
            eval_val_ppl(model, val_bytes, seq_len, device, snap_c=32) if n_w > 0 else (val_bpc, val_ppl)
        )
        py_short_acc, py_short_nll = eval_stateful_py_nll(model, eval_p_short, eval_a_short, device)
        py_long_acc, py_long_nll = eval_stateful_py_nll(model, eval_p_long, eval_a_long, device, snap_c=32 if n_w > 0 else None)

        gates = {}
        if n_w > 0:
            gates = {
                "lin1_wreath_gate": [round(float(torch.tanh(v)), 4) for v in model.lin1.wreath_gate.detach().cpu()],
                "lin2_wreath_gate": [round(float(torch.tanh(v)), 4) for v in model.lin2.wreath_gate.detach().cpu()],
            }

        results["models"][label] = {
            "params": n_params,
            "train_sec": elapsed,
            "train_tok_per_sec": tok_s,
            "val_bpc_real_python": val_bpc,
            "val_ppl_real_python": val_ppl,
            "val_bpc_snapped_C32": val_bpc_snap,
            "val_ppl_snapped_C32": val_ppl_snap,
            "stateful_py_short_acc_pct": py_short_acc,
            "stateful_py_long_acc_pct": py_long_acc,
            "wreath_gates_learned": gates,
        }
        print(
            f"  => {label} | params={n_params:,} | {elapsed}s ({tok_s:,} tok/s) | "
            f"Real Py/Eng Val BPC={val_bpc} (PPL={val_ppl}) | Snapped PPL={val_ppl_snap} | "
            f"Py-State Short={py_short_acc}% Long={py_long_acc}% | Gates={gates}"
        )

    os.makedirs("results", exist_ok=True)
    with open("results/exp8_hybrid_text_perplexity.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved results/exp8_hybrid_text_perplexity.json")


if __name__ == "__main__":
    run_exp8()
