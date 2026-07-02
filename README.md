# MYE023 – Parallel Systems, Assignment #2 (GPU)

Two problems parallelized on the GPU:

1. Matrix multiplication with CUDA
2. Sobel filter with OpenMP target offloading

Both exercises are compared against their serial versions and the speedup is measured.

Student: Athanasios Fourkiotis (ID 4940) — 2025–26
Execution system: parallax (NVIDIA Tesla P40)

## Files

| File | Description |
|---|---|
| `matmul_serial.c` | Serial matrix multiplication (provided) |
| `matrix-mul.c` | Exercise 1 — CUDA matmul |
| `sobel.c` | Exercise 2 — serial Sobel + two GPU versions |
| `run_experiments.sh` | Runs every scenario |
| `Amat{N}.txt`, `Bmat{N}.txt` | Matrix inputs (N = 512, 1024, 2048) |
| `Cmat{N}.txt` | Expected results for verification |
| `500.bmp`, `1000.bmp`, `1500.bmp` | Input images for Sobel |
| `results.txt` | Measurement output |
| `plot_*.png` | Charts |
| `Anafora2.pdf` | Written report |

## Exercise 1 — CUDA matrix multiplication

Computes `C = A · B` in CUDA for N = 512, 1024, 2048 and threads/block = 128, 256, 512.
The implementation is tiled with shared memory. GPU timing includes the
host↔device transfers. The result is verified against the serial version and against `Cmat<N>.txt`.

Compile & run:

```sh
nvcc -O2 -x cu -DN=1024 -DTHREADS=128 matrix-mul.c -o matrix-mul
./matrix-mul
```

Results:

| GPU times | Serial vs GPU |
|---|---|
| ![matmul times](plot_matmul_bars.png) | ![Serial vs GPU](plot_matmul_serial_vs_gpu.png) |

| Scaling with N | Speedup |
|---|---|
| ![matmul times](plot_matmul.png) | ![matmul speedup](plot_matmul_speedup.png) |

## Exercise 2 — Sobel filter (OpenMP target)

Applies Sobel edge detection to 24-bit BMP images on the GPU using OpenMP `target`.
There are two GPU versions: a combined one (`target teams distribute parallel for collapse(2)`)
and a nested one (separate `teams distribute` and `parallel for` directives). Three
images and several `num_teams` / `thread_limit` values are tried. The results are verified
pixel by pixel against the serial version.

Compile & run:

```sh
clang -O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
      -DNUM_TEAMS=30 -DTHREAD_LIMIT=128 sobel.c -lm -o sobel
./sobel 1500.bmp
```

Results:

| Times | Speedup |
|---|---|
| ![Sobel times](plot_sobel_times.png) | ![Sobel speedup](plot_sobel_speedup.png) |

## Running all the experiments

```sh
bash run_experiments.sh > results.txt 2>&1
```

The script runs every combination and writes the output to `results.txt`.

## Report

All the measurements, charts, and discussion of the results are in `Anafora2.pdf`.
