# MYE023 – Παράλληλα Συστήματα, Εργασία #2 (GPU)

Δύο προβλήματα παραλληλοποιημένα σε GPU:

1. Πολλαπλασιασμός πινάκων με CUDA
2. Sobel filter με OpenMP target offloading

Και στις δύο ασκήσεις γίνεται σύγκριση με τη σειριακή έκδοση και μέτρηση speedup.

Φοιτητής: Φουρκιώτης Αθανάσιος (ΑΜ 4940) — 2025–26
Σύστημα εκτέλεσης: parallax (NVIDIA Tesla P40)

## Αρχεία

| Αρχείο | Περιγραφή |
|---|---|
| `matmul_serial.c` | Σειριακός πολλαπλασιασμός πινάκων (δοσμένος) |
| `matrix-mul.c` | Άσκηση 1 — CUDA matmul |
| `sobel.c` | Άσκηση 2 — σειριακός Sobel + δύο GPU εκδόσεις |
| `run_experiments.sh` | Τρέχει όλα τα σενάρια |
| `Amat{N}.txt`, `Bmat{N}.txt` | Είσοδοι πινάκων (N = 512, 1024, 2048) |
| `Cmat{N}.txt` | Αναμενόμενα αποτελέσματα για επαλήθευση |
| `500.bmp`, `1000.bmp`, `1500.bmp` | Εικόνες εισόδου για τον Sobel |
| `results.txt` | Έξοδος των μετρήσεων |
| `plot_*.png` | Γραφικές |
| `Anafora2.pdf` | Γραπτή αναφορά |

## Άσκηση 1 — CUDA matrix multiplication

Υπολογισμός `C = A · B` σε CUDA για N = 512, 1024, 2048 και threads/block = 128, 256, 512.
Η υλοποίηση είναι tiled με shared memory. Στη χρονομέτρηση της GPU μετρώνται και οι
μεταφορές host↔device. Το αποτέλεσμα επαληθεύεται με τη σειριακή έκδοση και με τα `Cmat<N>.txt`.

Compile & run:

```sh
nvcc -O2 -x cu -DN=1024 -DTHREADS=128 matrix-mul.c -o matrix-mul
./matrix-mul
```

Αποτελέσματα:

| Χρόνοι (GPU) | Σειριακό vs GPU |
|---|---|
| ![Χρόνοι matmul](plot_matmul_bars.png) | ![Σειριακό vs GPU](plot_matmul_serial_vs_gpu.png) |

| Κλιμάκωση με το N | Speedup |
|---|---|
| ![matmul χρόνοι](plot_matmul.png) | ![matmul speedup](plot_matmul_speedup.png) |

## Άσκηση 2 — Sobel filter (OpenMP target)

Εφαρμογή Sobel edge-detection σε εικόνες BMP 24-bit στη GPU με OpenMP `target`.
Υπάρχουν δύο GPU εκδόσεις: μία combined (`target teams distribute parallel for collapse(2)`)
και μία nested (χωριστές οδηγίες `teams distribute` και `parallel for`). Δοκιμάζονται
3 εικόνες και διάφορες τιμές `num_teams` / `thread_limit`. Τα αποτελέσματα επαληθεύονται
pixel-pixel με τη σειριακή έκδοση.

Compile & run:

```sh
clang -O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
      -DNUM_TEAMS=30 -DTHREAD_LIMIT=128 sobel.c -lm -o sobel
./sobel 1500.bmp
```

Αποτελέσματα:

| Χρόνοι | Speedup |
|---|---|
| ![Sobel χρόνοι](plot_sobel_times.png) | ![Sobel speedup](plot_sobel_speedup.png) |

## Εκτέλεση όλων των πειραμάτων

```sh
bash run_experiments.sh > results.txt 2>&1
```

Το script τρέχει όλους τους συνδυασμούς και γράφει την έξοδο στο `results.txt`.

## Αναφορά

Όλες οι μετρήσεις, οι γραφικές και η συζήτηση των αποτελεσμάτων βρίσκονται στο `Anafora2.pdf`.
