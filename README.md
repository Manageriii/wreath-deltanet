# Dual-Gear Wreath-DeltaNet & Recursive Block Diffusion

**Breaking the Precision–Horizon–Expressivity Trilemma in Sub-Quadratic (`O(N)`) Sequence Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Hardware: Apple Silicon MPS / CUDA](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%7C%20CUDA-success.svg)](#reproducing-the-benchmarks)

> *Independent research built on a 24 GB Apple Silicon laptop in Finland. If this research or code is useful to your work, [GitHub Sponsors](.github/FUNDING.yml) go 100% toward funding a high-memory Apple M5 Ultra workstation to scale open-source `BF16`/`FP8` linear-attention and MLX Metal kernels to 30B–70B models.*

---

## Overview

Sub-quadratic sequence models (`Mamba-2/3`, `Gated DeltaNet-1/2`, `DeltaProduct`) face a fundamental **Precision–Horizon–Expressivity Trilemma**:
1. **Contractive recurrences (`|λ| < 1`)** stay numerically bounded in `BF16`/`FP8`, but collapse into `TC^0`—failing parity, modular clocks (`Z_m`), and non-abelian `NC^1` permutations (`S_5`).
2. **Isometric recurrences (`|λ| = 1`)** unlock `NC^1` expressivity in exact arithmetic, yet suffer catastrophic floating-point drift at finite horizons (`L* = Θ(2^p)` under Round-to-Nearest; `L* = Θ(2^{2p}/d)` under Stochastic Rounding).
3. **Token-level discrete FSMs (`NFSM`)** eliminate drift, but destroy intra-chunk linearity (`WY`/`SSD` GPU GEMMs) and continuous associative `O(d^2)` KV memory.

**Dual-Gear Wreath-DeltaNet** resolves all three simultaneously via Krohn-Rhodes semigroup decomposition:
* **Dual-Gear Worker 1 (Orthogonal Group Frame):** Combines `{-1, 0, +1}` signed-permutation **Swap-Boxes** (`L_4` Stiefel valley alignment) with **Unit-Circle `2D` Clock-Dials** (`1 - cos(Mθ)` alignment), evaluated linearly inside chunks of length `C` and snapped *only at chunk boundaries* (`C = 64` for `BF16`, `C = 4` for `FP8` E4M3) for provable **`L* = ∞`** drift immunity.
* **Clean Worker 2 (Wreath-Rotated Multi-Head `Gated DeltaNet`):** Co-rotates keys and queries in Worker 1's orthogonal prefix frame (`k_eff = U_t^T k_t`) with **5% Write & Address Dead-Zones** that eliminate distractor leakage and `FP8` cross-slot splash.

Full proofs, ablations, and theoretical derivations are in **[PAPER_DRAFT.md](PAPER_DRAFT.md)** and **[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md)**.

---

## Headline Results

| Experiment | Baseline (`BF16` / `FP8`) | **Dual-Gear Wreath-DeltaNet (Ours)** | Script |
| :--- | :---: | :---: | :--- |
| **Zero-Shot `BF16` Extrapolation (`L_train = 48 → L_eval = 32,768`, `682×`)** | `14.4%` *(Smooth Wreath)* / `62.5%` *(GDN)* | **`100.0%`** *(across ~26,200 distractors)* | `exp1_adaptive_locking.py`, `exp2_deadzone_gate.py` |
| **8-Bit `FP8` `S_5` Word Problem (`L = 16,384`)** | `0.0%` *(RTN)* / `2.3%` *(Stochastic Rounding)* | **`100.0%`** *(`C = 4` Chunk-Snapped)* | `exp5_fp8_and_clock_gears.py` |
| **8-Bit `FP8` `12`-Hour Clock & Write-Once Memory (`L = 8,192–16,384`)** | `40.6%` *(Clock)* / `12.5%` *(Write-Once)* | **`100.0%` / `100.0%`** | `exp5_fp8_and_clock_gears.py`, `exp6_trm_ncm_wreath.py` |
| **State-Conditioned Swaps (`L = 8,192`, Alternating Specialization)** | `61.7%` *(Standard / Naive 2-Layer)* | **`100.0%`** *(`20%` faster training)* | `exp3_conditional_two_layer.py` |
| **2-Hop Chained Conditionals (`1` Weight-Shared Block, 3 Passes)** | `46.9%` *(Pass 1)* | **`93.8%`** *(Pass 3, `+46.9` pts)* | `exp4b_refinement_v2.py` |
| **Apple Silicon (`MPS`) Tournament Chunk Scan (`L = 8,192`)** | `2,951 ms` *(Sequential)* | **`471 ms` (`6.3×` speedup, `100%` acc)** | `bench_gpu_chunks.py` |

---

## Reproducing the Benchmarks

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
```

---

## Citation

If you build upon Dual-Gear Wreath-DeltaNet or the Three-Regime Horizon Scaling Law, please cite **[PAPER_DRAFT.md](PAPER_DRAFT.md)**.
