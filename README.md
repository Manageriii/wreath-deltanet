# Dual-Gear Wreath-DeltaNet & Recursive Block Diffusion

**Breaking the Precision–Horizon–Expressivity Trilemma in Sub-Quadratic (`O(N)`) Sequence Models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24532887.svg)](https://doi.org/10.5281/zenodo.24532887)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Hardware: Apple Silicon MPS / CUDA](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%7C%20CUDA-success.svg)](#reproducing-the-benchmarks)

> *Independent research built on a 24 GB Apple Silicon laptop in Finland. If this research or code is useful to your work, [GitHub Sponsors](.github/FUNDING.yml) go 100% toward funding a high-memory Apple M5 Ultra workstation to scale open-source `BF16`/`FP8` linear-attention and MLX Metal kernels to 30B–70B models.*

---

## Visual Overview (`v1.1.0`)

![Figure 1: Breaking the BF16 & FP8 Precision-Horizon Trilemma](figures/fig1_horizon_scaling_bf16_fp8.svg)

![Figure 2: Apple Silicon MPS Chunk-Size Goldilocks Curve](figures/fig2_mps_chunk_goldilocks.svg)

![Figure 3: v1.1 Prime-Power CRT Universal 60-Tick Dials & 3:1 Hybrid Real Python Stdlib Check](figures/fig3_crt_clock_and_hybrid_lm.svg)

---

## Overview

Sub-quadratic sequence models (`Mamba-2/3`, `Gated DeltaNet-1/2`, `DeltaProduct`) face a fundamental **Precision–Horizon–Expressivity Trilemma**:
1. **Contractive recurrences (`|λ| < 1`)** stay numerically bounded in `BF16`/`FP8`, but collapse into `TC^0`—failing parity, modular clocks (`Z_m`), and non-abelian `NC^1` permutations (`S_5`).
2. **Isometric recurrences (`|λ| = 1`)** unlock `NC^1` expressivity in exact arithmetic, yet suffer catastrophic floating-point drift at finite horizons (`L* = Θ(2^p)` under Round-to-Nearest; `L* = Θ(2^{2p}/d)` under Stochastic Rounding).
3. **Token-level discrete FSMs (`NFSM`)** eliminate drift, but destroy intra-chunk linearity (`WY`/`SSD` GPU GEMMs) and continuous associative `O(d^2)` KV memory.

**Dual-Gear Wreath-DeltaNet** resolves all three simultaneously via Krohn-Rhodes semigroup decomposition:
* **Dual-Gear Worker 1 (Orthogonal Group Frame):** Combines `{-1, 0, +1}` signed-permutation **Swap-Boxes** (`L_4` Stiefel valley alignment) with **Prime-Power CRT Unit-Circle `2D` Clock-Dials** (`Z_4 × Z_3 × Z_5 ≅ Z_60`, `90° / 120° / 72°`), evaluated linearly inside chunks of length `C` and snapped *only at chunk boundaries* (`C = 64` for `BF16`, `C = 4` for `FP8` E4M3) for provable **`L* = ∞`** drift immunity.
* **Clean Worker 2 (Wreath-Rotated Multi-Head `Gated DeltaNet`):** Co-rotates keys and queries in Worker 1's orthogonal prefix frame (`k_eff = U_t^T k_t`) with **5% Write & Address Dead-Zones** that eliminate distractor leakage and `FP8` cross-slot splash.

Full proofs, ablations, and theoretical derivations are in **[PAPER_DRAFT.md](PAPER_DRAFT.md)** and **[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md)**.

---

## Headline Results (`v1.0` + New `v1.1` Experiments)

| Experiment | Baseline (`BF16` / `FP8`) | **Dual-Gear Wreath-DeltaNet (Ours)** | Script |
| :--- | :---: | :---: | :--- |
| **Zero-Shot `BF16` Extrapolation (`L_train = 48 → L_eval = 32,768`, `682×`)** | `14.4%` *(Smooth Wreath)* / `62.5%` *(GDN)* | **`100.0%`** *(across ~26,200 distractors)* | `exp1_adaptive_locking.py`, `exp2_deadzone_gate.py` |
| **8-Bit `FP8` `S_5` Word Problem (`L = 16,384`)** | `0.0%` *(RTN)* / `2.3%` *(Stochastic Rounding)* | **`100.0%`** *(`C = 4` Chunk-Snapped)* | `exp5_fp8_and_clock_gears.py` |
| **8-Bit `FP8` `12`-Hour Clock & Write-Once Memory (`L = 8,192–16,384`)** | `40.6%` *(Clock)* / `12.5%` *(Write-Once)* | **`100.0%` / `100.0%`** | `exp5_fp8_and_clock_gears.py`, `exp6_trm_ncm_wreath.py` |
| **State-Conditioned Swaps (`L = 8,192`, Alternating Specialization)** | `61.7%` *(Standard / Naive 2-Layer)* | **`100.0%`** *(`20%` faster training)* | `exp3_conditional_two_layer.py` |
| **2-Hop Chained Conditionals (`1` Weight-Shared Block, 3 Passes)** | `46.9%` *(Pass 1)* | **`93.8%`** *(Pass 3, `+46.9` pts)* | `exp4b_refinement_v2.py` |
| **Apple Silicon (`MPS`) Tournament Chunk Scan (`L = 8,192`)** | `2,951 ms` *(Sequential)* | **`471 ms` (`6.3×` speedup, `100%` acc)** | `bench_gpu_chunks.py` |
| **`v1.1` Universal 60-Tick Clock via Prime-Power CRT (`Z_4 × Z_3 × Z_5`, `L = 16,384`, 0-reset)** | `0.0%` `BF16` / `7.8%` `FP8` *(Unsnapped)* · `2.3%` *(Mono `Z_60` `FP8`)* | **`100.0%` `BF16` (`C=64`) & `100.0%` `FP8` (`C=4`) across ALL 8 moduli (`mod 2,3,4,5,6,10,12,60`)** | `exp7_universal_60tick_dial.py` |
| **`v1.1` `3:1` Hybrid Real Python Stdlib LM (4 Unseen Modules: `difflib`, `statistics`, `textwrap`, `csv`)** | `Val PPL = 9.897` (`13,853 tok/s`) | **`Val PPL = 10.181` (`10.182` snapped `C=32`, `+0.001` snap delta, `+0.25%` params, `13,548 tok/s`)** *(Honest finding: short byte-level prose keeps zero-init gates near Identity `|g| < 0.006` — see §4.7)* | `exp8_hybrid_text_perplexity.py` |

---

## Reproducing the Benchmarks & Figures

```bash
pip install torch numpy

# 1. Adaptive performance-gated valley locking & 682x BF16 extrapolation
python3 exp1_adaptive_locking.py
python3 exp2_deadzone_gate.py

# 2. Alternating layer specialization & weight-shared 3-pass TRM refinement
python3 exp3_conditional_two_layer.py
python3 exp4b_refinement_v2.py

# 3. 8-bit FP8 chunk scaling law (C=4 vs C=64) & 2D 30-degree Clock-Dials
python3 exp5_fp8_and_clock_gears.py
python3 exp6_trm_ncm_wreath.py

# 4. Apple Silicon MPS / GPU chunk-parallel tournament scan benchmark
python3 bench_gpu_chunks.py

# 5. Stage 2: Unsupervised SuperBPE + CoBPE Block-Diffusion Language Model
python3 stage2/run_block_diffusion_stage2.py

# 6. v1.1: Universal 60-Tick Prime-Power CRT Dials (Z_4 x Z_3 x Z_5) & 3:1 Hybrid Real Python LM
python3 exp7_universal_60tick_dial.py
python3 exp8_hybrid_text_perplexity.py

# 7. Regenerate all 3 vector SVG figures from live JSON results (zero extra dependencies!)
python3 generate_figures.py
```

---

## Citation

If you build upon Dual-Gear Wreath-DeltaNet or the Three-Regime Horizon Scaling Law, please cite the archived Zenodo record ([`10.5281/zenodo.24532887`](https://doi.org/10.5281/zenodo.24532887)) and **[PAPER_DRAFT.md](PAPER_DRAFT.md)**:

```bibtex
@software{markus_2026_wreath_deltanet,
  author       = {Markus (Manageriii)},
  title        = {Dual-Gear Wreath-DeltaNet \& Recursive Block Diffusion: Breaking the Precision-Horizon-Expressivity Trilemma in Sub-Quadratic Sequence Models},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.1.0},
  doi          = {10.5281/zenodo.24532887},
  url          = {https://doi.org/10.5281/zenodo.24532887}
}
```
