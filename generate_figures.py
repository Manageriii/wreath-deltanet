#!/usr/bin/env python3
"""Generate publication-grade vector SVG figures for Dual-Gear Wreath-DeltaNet (v1.1).

Zero external dependencies (pure Python 3 stdlib).
Outputs crisp white-card SVGs (readable on both GitHub Light & Dark themes and
directly embeddable into PDF/LaTeX reports for TEK Kipina 2027 / arXiv):
  - figures/fig1_horizon_scaling_bf16_fp8.svg
  - figures/fig2_mps_chunk_goldilocks.svg
  - figures/fig3_crt_clock_and_hybrid_lm.svg
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_wrap(w: int, h: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif;">\n'
        f'  <rect x="1" y="1" width="{w-2}" height="{h-2}" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="2"/>\n'
        f"{body}\n"
        f"</svg>\n"
    )


def draw_line_panel(
    x0: float,
    y0: float,
    pw: float,
    ph: float,
    title: str,
    subtitle: str,
    x_labels: list[str],
    series: list[dict],
    y_label: str = "Accuracy (%)",
    legend_col2_offset: float = 238.0,
) -> str:
    max_y = 106.0  # slight headroom so 100% markers sit cleanly inside top border
    parts = [
        f'  <text x="{x0 + pw/2:.1f}" y="{y0 - 34:.1f}" text-anchor="middle" font-size="14.8" font-weight="700" fill="#0f172a">{esc(title)}</text>',
        f'  <text x="{x0 + pw/2:.1f}" y="{y0 - 15:.1f}" text-anchor="middle" font-size="11" fill="#475569">{esc(subtitle)}</text>',
        f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>',
    ]

    # Y gridlines 0..100
    for yv in (0, 20, 40, 60, 80, 100):
        yy = y0 + ph - (yv / max_y) * ph
        parts.append(f'  <line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'  <text x="{x0 - 8:.1f}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="#475569">{yv}%</text>')

    # X ticks with slight horizontal padding inside plot box so edge dots stay crisp
    pad_x = 16.0
    n = len(x_labels)
    xs = [x0 + pad_x + (i / max(1, n - 1)) * (pw - 2 * pad_x) for i in range(n)]
    for xi, lbl in zip(xs, x_labels):
        parts.append(f'  <line x1="{xi:.1f}" y1="{y0}" x2="{xi:.1f}" y2="{y0+ph}" stroke="#f1f5f9" stroke-width="1"/>')
        parts.append(f'  <text x="{xi:.1f}" y="{y0 + ph + 18:.1f}" text-anchor="middle" font-size="11" fill="#334155">{esc(lbl)}</text>')

    parts.append(
        f'  <text x="{x0 - 46:.1f}" y="{y0 + ph/2:.1f}" text-anchor="middle" font-size="12" font-weight="600" '
        f'fill="#334155" transform="rotate(-90 {x0 - 46:.1f} {y0 + ph/2:.1f})">{esc(y_label)}</text>'
    )
    parts.append(
        f'  <text x="{x0 + pw/2:.1f}" y="{y0 + ph + 39:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">Sequence Length L (log scale)</text>'
    )

    # Lines & points
    for s in series:
        pts = []
        valid_pts = []
        for xi, val in zip(xs, s["values"]):
            if val is None:
                continue
            yy = y0 + ph - (val / max_y) * ph
            pts.append(f"{xi:.1f},{yy:.1f}")
            valid_pts.append((xi, yy, val))
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        sw = s.get("width", 2.6)
        parts.append(
            f'  <polyline fill="none" stroke="{s["color"]}" stroke-width="{sw}"{dash} points="{" ".join(pts)}"/>'
        )
        for xi, yy, _ in valid_pts:
            r = 4.2 if sw >= 3.0 else 3.4
            parts.append(f'  <circle cx="{xi:.1f}" cy="{yy:.1f}" r="{r}" fill="{s["color"]}" stroke="#ffffff" stroke-width="1.2"/>')

    # Legend below panel (explicit wide two-column separation so labels never touch)
    ly = y0 + ph + 62
    lx = x0
    for idx, s in enumerate(series):
        col = idx % 2
        row = idx // 2
        cx = lx + col * legend_col2_offset
        cy = ly + row * 21
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        parts.append(f'  <line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx+18:.1f}" y2="{cy:.1f}" stroke="{s["color"]}" stroke-width="3"{dash}/>')
        parts.append(f'  <text x="{cx+24:.1f}" y="{cy+4:.1f}" font-size="10.8" fill="#1e293b">{esc(s["label"])}</text>')

    return "\n".join(parts)


def make_fig1():
    w, h = 1120, 495
    p1 = draw_line_panel(
        x0=76,
        y0=102,
        pw=452,
        ph=250,
        title="(A) Low-Precision Horizon Collapse vs. Chunk Snap (S5)",
        subtitle="Exact Householder generators in dense rotated basis (exp5)",
        x_labels=["64", "256", "1,024", "4,096", "16,384"],
        series=[
            {"label": "BF16 C=64 & FP8 C=4 (Ours)", "values": [100.0, 100.0, 100.0, 100.0, 100.0], "color": "#059669", "width": 3.5},
            {"label": "BF16 RTN (DeltaProduct-style)", "values": [100.0, 100.0, 100.0, 96.1, 0.8], "color": "#2563eb", "dash": "6,4"},
            {"label": "FP8 Stoch. Round + Stiefel", "values": [100.0, 100.0, 100.0, 32.0, 2.3], "color": "#d97706", "dash": "4,3"},
            {"label": "FP8 Chunk C=256 (Too Wide)", "values": [100.0, 89.8, 70.3, 22.7, 0.0], "color": "#9333ea", "dash": "2,3"},
            {"label": "FP8 RTN Unsnapped", "values": [100.0, 89.8, 1.6, 0.0, 0.0], "color": "#dc2626"},
        ],
        legend_col2_offset=234.0,
    )
    p2 = draw_line_panel(
        x0=618,
        y0=102,
        pw=452,
        ph=250,
        title="(B) Zero-Shot 682x Extrapolation (Trained ONLY at L=48)",
        subtitle="80% distractor chatter (~26,200 fillers at L=32,768; exp1 & exp2)",
        x_labels=["48 (1x)", "2,048 (42x)", "8,192 (170x)", "32,768 (682x)"],
        series=[
            {"label": "Adaptive Magnet + Dead-Zone (Ours)", "values": [100.0, 99.4, 100.0, 100.0], "color": "#059669", "width": 3.5},
            {"label": "Adaptive Magnet + Leaky Gate", "values": [100.0, 100.0, 100.0, 99.4], "color": "#0284c7", "dash": "5,3"},
            {"label": "Baseline Gated DeltaNet (No Wreath)", "values": [63.7, 60.0, 62.5, None], "color": "#64748b", "dash": "4,4"},
            {"label": "Smooth Wreath-GDN (No Magnet)", "values": [100.0, 18.1, 10.6, 14.4], "color": "#dc2626"},
        ],
        legend_col2_offset=254.0,
    )
    header = f'  <text x="{w/2:.1f}" y="34" text-anchor="middle" font-size="18.5" font-weight="800" fill="#0f172a">Figure 1: Breaking the BF16 &amp; FP8 Precision-Horizon Trilemma</text>'
    svg = svg_wrap(w, h, header + "\n" + p1 + "\n" + p2)
    (FIG_DIR / "fig1_horizon_scaling_bf16_fp8.svg").write_text(svg, encoding="utf-8")


def make_fig2():
    # Apple Silicon MPS Goldilocks curve (bench_gpu_chunks.py, L=8,192)
    w, h = 960, 455
    x0, y0, pw, ph = 94, 102, 764, 240
    cats = [
        "Sequential\n(Loop)",
        "Chunk C=8\n(1024 chunks)",
        "Chunk C=64\n(OURS: Sweet Spot)",
        "Chunk C=512\n(16 chunks)",
        "Unsnapped\n(C=8,192)",
    ]
    ms_vals = [2951.0, 1063.0, 471.0, 560.0, 916.0]
    acc_vals = [100.0, 100.0, 100.0, 96.0, 42.5]

    parts = [
        f'  <text x="{w/2:.1f}" y="34" text-anchor="middle" font-size="18" font-weight="800" fill="#0f172a">Figure 2: Apple Silicon (M4 Pro MPS) Chunk-Size Goldilocks Curve (L = 8,192)</text>',
        f'  <text x="{w/2:.1f}" y="58" text-anchor="middle" font-size="12" fill="#475569">Tournament-bracket chunk scan achieves 6.3x wall-clock speedup at C=64 while preserving 100.0% BF16 accuracy</text>',
        f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>',
    ]

    # Left Y axis: ms (0..4000) keeps 2951 ms bar well below the 100% accuracy line
    max_ms = 4000.0
    for mv in (0, 1000, 2000, 3000, 4000):
        yy = y0 + ph - (mv / max_ms) * ph
        parts.append(f'  <line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'  <text x="{x0-10}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#1d4ed8">{mv} ms</text>')

    # Right Y axis: Acc (0..100%) aligned with gridlines and matched to red curve color
    for av in (0, 25, 50, 75, 100):
        yy = y0 + ph - (av / 100.0) * ph
        parts.append(f'  <text x="{x0+pw+10}" y="{yy+4:.1f}" text-anchor="start" font-size="11" fill="#b91c1c">{av}%</text>')

    parts.append(
        f'  <text x="24" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="12" font-weight="700" '
        f'fill="#1d4ed8" transform="rotate(-90 24 {y0+ph/2:.1f})">GPU Wall-Clock Latency (ms, lower is better)</text>'
    )
    parts.append(
        f'  <text x="{w-24}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="12" font-weight="700" '
        f'fill="#b91c1c" transform="rotate(90 {w-24} {y0+ph/2:.1f})">BF16 State Accuracy (%)</text>'
    )

    step_w = pw / len(cats)
    bar_w = step_w * 0.46
    acc_pts = []
    for i, (cat, ms, acc) in enumerate(zip(cats, ms_vals, acc_vals)):
        cx = x0 + (i + 0.5) * step_w
        bh = (ms / max_ms) * ph
        by = y0 + ph - bh
        is_best = i == 2
        fill = "#059669" if is_best else ("#94a3b8" if i == 0 else "#3b82f6")
        parts.append(f'  <rect x="{cx - bar_w/2:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="6" fill="{fill}" opacity="0.88"/>')
        lbl_top = f"{int(ms)} ms (6.3x!)" if is_best else f"{int(ms)} ms"
        parts.append(f'  <text x="{cx:.1f}" y="{by - 8:.1f}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">{lbl_top}</text>')
        for li, line in enumerate(cat.split("\n")):
            fw = "700" if is_best else "500"
            parts.append(f'  <text x="{cx:.1f}" y="{y0 + ph + 19 + li*15:.1f}" text-anchor="middle" font-size="11" font-weight="{fw}" fill="#334155">{esc(line)}</text>')
        ay = y0 + ph - (acc / 100.0) * ph
        acc_pts.append((cx, ay, acc))

    parts.append(
        '  <polyline fill="none" stroke="#dc2626" stroke-width="3" stroke-dasharray="6,3" points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in acc_pts)
        + '"/>'
    )
    for idx, (x, y, acc) in enumerate(acc_pts):
        parts.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#dc2626" stroke="#ffffff" stroke-width="1.5"/>')
        if idx < 4:
            # Place 100.0% and 96.0% cleanly ABOVE the red curve (well inside headroom above plot box)
            parts.append(
                f'  <text x="{x:.1f}" y="{y - 11:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#b91c1c">{acc:.1f}%</text>'
            )
        else:
            # Place 42.5% cleanly to the RIGHT of the dot, clear of both the incoming diagonal and the 916 ms bar
            parts.append(
                f'  <text x="{x + 11:.1f}" y="{y + 4:.1f}" text-anchor="start" font-size="11" font-weight="700" fill="#b91c1c">{acc:.1f}%</text>'
            )

    # Legend centered at bottom
    parts.append(f'  <rect x="215" y="{h-36}" width="14" height="14" rx="3" fill="#059669"/>')
    parts.append(f'  <text x="235" y="{h-25}" font-size="11.5" fill="#1e293b">Optimal Chunk C=64 (471 ms, 6.3x speedup, 100% acc)</text>')
    parts.append(f'  <line x1="575" y1="{h-29}" x2="599" y2="{h-29}" stroke="#dc2626" stroke-width="3" stroke-dasharray="6,3"/>')
    parts.append(f'  <text x="605" y="{h-25}" font-size="11.5" fill="#1e293b">BF16 State Accuracy (%)</text>')

    svg = svg_wrap(w, h, "\n".join(parts))
    (FIG_DIR / "fig2_mps_chunk_goldilocks.svg").write_text(svg, encoding="utf-8")


def make_fig3():
    exp7 = json.loads((RES_DIR / "exp7_universal_60tick.json").read_text(encoding="utf-8"))
    exp8 = json.loads((RES_DIR / "exp8_hybrid_text_perplexity.json").read_text(encoding="utf-8"))

    w, h = 1120, 495
    parts = [
        f'  <text x="{w/2:.1f}" y="34" text-anchor="middle" font-size="18" font-weight="800" fill="#0f172a">Figure 3: v1.1 Results — Prime-Power CRT Universal Clocks &amp; 3:1 Hybrid Real Python/English LM</text>'
    ]

    # Left panel: Exp 7 CRT vs Monolithic Z_60 in 8-bit FP8 + End-to-End 0-Reset Extrapolation
    x0, y0, pw, ph = 80, 106, 448, 250
    max_acc = 115.0  # headroom so 100.0% bar labels sit cleanly inside plot box, never touching subtitle
    parts += [
        f'  <text x="{x0+pw/2:.1f}" y="{y0-34:.1f}" text-anchor="middle" font-size="14.8" font-weight="700" fill="#0f172a">(A) Universal 60-Tick Clock: Monolithic vs. CRT (Exp 7)</text>',
        f'  <text x="{x0+pw/2:.1f}" y="{y0-15:.1f}" text-anchor="middle" font-size="11" fill="#475569">FP8 Exact (left 2) &amp; End-to-End Zero-Reset at L=16,384 (right 3)</text>',
        f'  <rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>',
    ]
    for yv in (0, 20, 40, 60, 80, 100):
        yy = y0 + ph - (yv / max_acc) * ph
        parts.append(f'  <line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'  <text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#475569">{yv}%</text>')
    parts.append(
        f'  <text x="{x0-48}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="11.5" font-weight="600" '
        f'fill="#334155" transform="rotate(-90 {x0-48} {y0+ph/2:.1f})">Exact mod-60 Accuracy (%)</text>'
    )

    fp8_mono_c4 = exp7["part_7a_margin_sweep"]["fp8"]["C=4"]["16384"]
    fp8_crt_c4 = exp7["part_7a_margin_sweep"]["fp8_crt_z12xz5"]["C=4"]["16384"]
    d7 = exp7.get("part_7d_prime_power_crt_z4xz3xz5", {})
    p7d_snap = d7.get("prime_crt_z4xz3xz5_snapped", {})
    p7d_bf16_16k = p7d_snap.get("bf16_extrapolation", {}).get("16384", 0.0)
    p7d_fp8_16k = p7d_snap.get("fp8_extrapolation", {}).get("16384", 0.0)
    p7d_smooth_16k = d7.get("prime_crt_smooth_unsnapped", {}).get("bf16_extrapolation", {}).get("16384", 0.0)

    bars_a = [
        ("Mono Z60\nFP8 C=4\n(16K)", fp8_mono_c4, "#dc2626"),
        ("CRT Z12xZ5\nFP8 C=4\n(16K)", fp8_crt_c4, "#059669"),
        ("Trained\nUnsnap\n(16K 0-rst)", p7d_smooth_16k, "#f97316"),
        ("Trained\nCRT BF16\n(16K 0-rst)", p7d_bf16_16k, "#2563eb"),
        ("Trained\nCRT FP8\n(16K 0-rst)", p7d_fp8_16k, "#0d9488"),
    ]
    sw = pw / len(bars_a)
    bw = sw * 0.56
    for i, (lbl, val, col) in enumerate(bars_a):
        cx = x0 + (i + 0.5) * sw
        bh = max(2.5, (val / max_acc) * ph)
        by = y0 + ph - bh
        parts.append(f'  <rect x="{cx - bw/2:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="5" fill="{col}" opacity="0.9"/>')
        parts.append(f'  <text x="{cx:.1f}" y="{by - 8:.1f}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">{val:.1f}%</text>')
        for li, ln in enumerate(lbl.split("\n")):
            parts.append(f'  <text x="{cx:.1f}" y="{y0 + ph + 17 + li*14:.1f}" text-anchor="middle" font-size="10.4" fill="#334155">{esc(ln)}</text>')

    # Right panel: Exp 8 Hybrid Real Python/English Val PPL
    rx0, ry0, rpw, rph = 618, 106, 448, 250
    m_base = exp8["models"]["Hybrid-DeltaNet (8/8 Standard)"]
    m_ours = exp8["models"]["Hybrid-Wreath-DeltaNet (2/8 Wreath + 6/8 Std)"]
    parts += [
        f'  <text x="{rx0+rpw/2:.1f}" y="{ry0-34:.1f}" text-anchor="middle" font-size="14.8" font-weight="700" fill="#0f172a">(B) 3:1 Hybrid LM on Unseen Python Stdlib (Exp 8)</text>',
        f'  <text x="{rx0+rpw/2:.1f}" y="{ry0-15:.1f}" text-anchor="middle" font-size="11" fill="#475569">Held-out modules: difflib, statistics, textwrap, csv (187 KB; +0.25% params)</text>',
        f'  <rect x="{rx0}" y="{ry0}" width="{rpw}" height="{rph}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>',
    ]
    max_ppl = max(12.0, math.ceil(max(m_base["val_ppl_real_python"], m_ours["val_ppl_real_python"]) + 1.5))
    for pv in (0, 3, 6, 9, 12):
        if pv > max_ppl:
            continue
        yy = ry0 + rph - (pv / max_ppl) * rph
        parts.append(f'  <line x1="{rx0}" y1="{yy:.1f}" x2="{rx0+rpw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'  <text x="{rx0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#475569">{pv}</text>')
    parts.append(
        f'  <text x="{rx0-40}" y="{ry0+rph/2:.1f}" text-anchor="middle" font-size="11.5" font-weight="600" '
        f'fill="#334155" transform="rotate(-90 {rx0-40} {ry0+rph/2:.1f})">Validation Perplexity (lower is better)</text>'
    )

    bars_b = [
        ("Hybrid-DeltaNet\n(8/8 Standard)\nVal PPL", m_base["val_ppl_real_python"], "#64748b"),
        ("Hybrid-Wreath\n(2/8 Wreath)\nVal PPL", m_ours["val_ppl_real_python"], "#059669"),
        ("Hybrid-Wreath\n(C=32 Snapped)\nVal PPL", m_ours["val_ppl_snapped_C32"], "#0284c7"),
    ]
    swb = rpw / len(bars_b)
    bwb = swb * 0.48
    for i, (lbl, val, col) in enumerate(bars_b):
        cx = rx0 + (i + 0.5) * swb
        bh = (val / max_ppl) * rph
        by = ry0 + rph - bh
        parts.append(f'  <rect x="{cx - bwb/2:.1f}" y="{by:.1f}" width="{bwb:.1f}" height="{bh:.1f}" rx="6" fill="{col}" opacity="0.9"/>')
        parts.append(f'  <text x="{cx:.1f}" y="{by - 8:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">{val:.3f}</text>')
        for li, ln in enumerate(lbl.split("\n")):
            parts.append(f'  <text x="{cx:.1f}" y="{ry0 + rph + 17 + li*14:.1f}" text-anchor="middle" font-size="10.5" fill="#334155">{esc(ln)}</text>')

    # Footer caption summarizing Exp 8 gates & speed
    foot = (
        f'Exp 8 Speed: DeltaNet {m_base["train_tok_per_sec"]:,} tok/s vs. Wreath {m_ours["train_tok_per_sec"]:,} tok/s '
        f'(+0.25% params) | Learned Wreath gates: L1={m_ours["wreath_gates_learned"].get("lin1_wreath_gate")}, '
        f'L2={m_ours["wreath_gates_learned"].get("lin2_wreath_gate")}'
    )
    parts.append(f'  <text x="{w/2:.1f}" y="{h - 22}" text-anchor="middle" font-size="11" fill="#475569">{esc(foot)}</text>')

    svg = svg_wrap(w, h, "\n".join(parts))
    (FIG_DIR / "fig3_crt_clock_and_hybrid_lm.svg").write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_fig1()
    make_fig2()
    make_fig3()
    print("Generated all 3 vector SVG figures in", FIG_DIR)
