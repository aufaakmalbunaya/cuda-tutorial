# Tutorial GPU: CUDA C++

Hands-on GPU programming in CUDA C++ — 8 exercises plus a mini-project
(2D five-point stencil), with a full lab report in Indonesian.

The local VM has no GPU, so everything was compiled with nvcc 12.8
(`-std=c++17 -O3 -lineinfo -arch=native`) and actually run on a Kaggle
notebook with a **Tesla T4** (compute capability 7.5, 40 SMs, 15 GiB).
`cuda_tutorial_kaggle.ipynb` is the notebook used; `evidence/` holds the
executed notebook outputs and the raw measurement CSVs (77 runs);
`latex_report/` holds the Indonesian report (`.tex` + compiled `.pdf`).

## Layout

| Path | Contents |
|---|---|
| `src/` | 13 CUDA programs: 8 tutorial exercises, modifications, mini-project |
| `cuda_tutorial_kaggle.ipynb` | Kaggle notebook (import → set Accelerator=GPU → Run All) |
| `evidence/` | executed-notebook outputs (`.txt`) + raw measurements (`.csv`) |
| `latex_report/` | Indonesian lab report (`.tex` + compiled `.pdf`) + figures |
| `Tutorial_GPU_Cuda.pdf` | the tutorial document this work follows |
| `tugas_cuda_Aufa_Akmal_Bunaya_overleaf.zip` | Overleaf submission bundle |

## Quick start (Kaggle, GPU accelerator)

1. Upload `cuda_tutorial_kaggle.ipynb` via **File → Import Notebook**.
2. Set **Accelerator → GPU** in the right sidebar.
3. **Run All**, then **File → Download as .ipynb**.

Or compile one program directly on any CUDA machine:

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=native src/vector_add.cu -o vector_add
./vector_add
```

## Key results (Tesla T4, medians)

- **Exercise 4** (vector add, n=10⁷): kernel-only speedup **19.1×** vs CPU,
  but transfer-inclusive operation speedup only **0.32×** — PCIe transfer
  dominates unless data stays resident on the device.
- **Exercise 5** (grid-stride, n=10⁶): best launch 256 blocks × 128 threads →
  **232.6 GB/s** effective bandwidth; worst 64×64 → 72.2 GB/s (3.2× gap).
- **Exercise 7** (matmul): tiled 16×16 beats naive by ~1.5× —
  N=512: **482.6 vs 312.1 GFLOP/s**, all 30 runs error-free.
- **Exercise 8**: `compute-sanitizer` clean on correct programs; caught an
  out-of-bounds write (97 errors, launch failure) and a missing
  `__syncthreads()` (racecheck hazard, wrong reduction result).
- **Mini-project** (2D stencil, 1024² × 20 steps): resident kernel 0.94 ms
  vs CPU 50.6 ms (54×); per-step transfers 36.0 ms — 12× slower than resident.
