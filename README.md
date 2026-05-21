# MYE023 – Παράλληλα Συστήματα και Προγραμματισμός — Εργασία #2 (GPU)

Παραλληλοποίηση δύο προβλημάτων σε GPU: πολλαπλασιασμός πινάκων με **CUDA**
και Sobel filter με **OpenMP target offloading**. Σύγκριση των GPU εκδόσεων
με τη σειριακή και μέτρηση speedup.

**Φοιτητής:** Φουρκιώτης Αθανάσιος (ΑΜ 4940)
**Ακαδημαϊκό έτος:** 2025–26 · **Διδάσκων:** Βασίλειος Δημακόπουλος
**Σύστημα εκτέλεσης:** parallax (NVIDIA Tesla P40, 3840 CUDA cores)

## Περιεχόμενα

| Αρχείο | Περιγραφή |
|---|---|
| `matmul_serial.c` | Σειριακός πολλαπλασιασμός πινάκων (δοσμένος από τη διδασκαλία) |
| `matrix-mul.c` | **Άσκηση 1** — CUDA tiled matmul με shared memory |
| `sobel.c` | **Άσκηση 2** — σειριακός Sobel + δύο GPU εκδόσεις με OpenMP target |
| `run_experiments.sh` | Script που τρέχει όλα τα σενάρια (N × THREADS, image × config) |
| `Amat{N}.txt`, `Bmat{N}.txt` | Είσοδοι πινάκων για N ∈ {512, 1024, 2048} |
| `Cmat{N}.txt` | Αναμενόμενα αποτελέσματα για επαλήθευση |
| `500.bmp`, `1000.bmp`, `1500.bmp` | Εικόνες εισόδου για τον Sobel |
| `results.txt` | Raw έξοδος όλων των μετρήσεων |
| `plot_matmul*.png`, `plot_sobel*.png` | Γραφικές: χρόνοι, speedup, σύγκριση σειριακό vs GPU |
| `Anafora2.pdf` | Γραπτή αναφορά με αναλύσεις, γραφικές και συμπεράσματα |

## Άσκηση 1 — CUDA matrix multiplication (50%)

**Τι ζητάει:** υλοποίηση του `C = A · B` σε CUDA για N ∈ {512, 1024, 2048}
και threads/block ∈ {128, 256, 512}. Σύγκριση με τη σειριακή έκδοση,
χρονομέτρηση **συμπεριλαμβανομένων** των μεταφορών host↔device, και
επαλήθευση ορθότητας με τα δοσμένα `Cmat<N>.txt`.

**Πώς το σκέφτηκα:**
Αρχικά **Tiled με shared memory αντί naive.** Οπότε ξεκίνησα από μια naive έκδοση όπου
  κάθε thread διάβαζε μόνο του από global memory — ήταν πολύ αργή. Πέρασα σε
  tiled, με δύο shared arrays (`As`, `Bs`) μεγέθους `TILE_SIZE×TILE_SIZE`,
  ώστε κάθε στοιχείο του A/B να διαβάζεται από την global memory **μία φορά
  ανά tile** και να επαναχρησιμοποιείται από όλα τα threads του block. Μετά με **Block geometry — 2D 
  με σταθερό πλάτος.** Tο block είναι `TILE_SIZE × BLOCK_ROWS`,
  όπου `BLOCK_ROWS = THREADS / TILE_SIZE`. Με `TILE_SIZE=16` παίρνω 8/16/32
  γραμμές για 128/256/512 threads/block αντίστοιχα. Έτσι το ίδιο kernel
  παραμετροποιείται από compile-time flags (`-DN=...`, `-DTHREADS=...`).
 **Φόρτωση του B με εσωτερικό loop.** Επειδή το tile του B είναι πάντα
  16×16 αλλά το ύψος του block αλλάζει με τα threads, η φόρτωση του `Bs`
  γίνεται με loop `load_y += BLOCK_ROWS` ώστε να δουλεύει σωστά για όλες
  τις διαμορφώσεις. **`__syncthreads()` ανάμεσα σε φόρτωση και υπολογισμό**, και ένα δεύτερο
  πριν το επόμενο tile, για να μη χρησιμοποιεί κανένα thread παλιά shared
  data ενώ άλλο γράφει νέα. Επίσης έχω **Χρονομέτρηση δίκαιη.** Η σειριακή τρέχει 1 φορά (στο N=2048 αργεί
  αρκετά). Η GPU τρέχει 1 warm-up (δεν μετράει — αφορά JIT/init κόστος)
  και 4 timed runs. Η μεταφορά H2D και D2H **μετράνε** στον GPU χρόνο
 Τέλος η **Διπλή επαλήθευση:** το αποτέλεσμα της GPU συγκρίνεται κατευθείαν με
  τον σειριακό υπολογισμό **και** με το `Cmat<N>.txt`.

**Compile & run:**

```sh
nvcc -O2 -x cu -DN=1024 -DTHREADS=128 matrix-mul.c -o matrix-mul
./matrix-mul
```

## Άσκηση 2 — Sobel filter με OpenMP target offloading (50%)

**Τι ζητάει:** εφαρμογή του Sobel edge-detection σε εικόνες BMP 24-bit
στη GPU, μέσω OpenMP `target` directives. Σύγκριση δύο τρόπων κατανομής
δουλειάς (combined vs nested directives), για 3 εικόνες (500/1000/1500
pixels) και διάφορες παραμέτρους `num_teams`/`thread_limit`.

**Πώς το σκέφτηκα:**

- **Δύο GPU εκδόσεις.**
  - `sobel_omp_device` (**combined**): μία οδηγία `target teams distribute
    parallel for collapse(2)`. Τα δύο loops (γραμμές × στήλες) ενώνονται
    σε έναν χώρο iterations και ο compiler μοιράζει όλα τα pixels στα
    teams/threads.
  - `sobel_omp_device_nested` (**nested**): χωριστά `target` → `teams
    distribute` (γραμμές στα teams) → `parallel for` (στήλες στα threads).
- **Mapping δεδομένων.** Τα κανάλια εισόδου (`in_r/g/b`) και τα Sobel
  kernels (`Gx`, `Gy`) μπαίνουν με `map(to:)` — μόνο διαβάζονται. Τα κανάλια
  εξόδου με `map(from:)` — μόνο γράφονται και επιστρέφουν στον host.
- **Επιλογή `num_teams` και `thread_limit`.** Το `run_experiments.sh`
  δοκιμάζει 15 ζευγάρια από `(32, 120)` μέχρι `(1024, 4)`, ώστε τα `teams
  × thread_limit` να καλύπτουν τους 3840 cores της P40. Τα thread limits
  είναι **πολλαπλάσια του 32** επειδή το warp της NVIDIA έχει 32 threads
  — διαφορετικά υπάρχουν idle lanes.
- **Race conditions.** Οι μεταβλητές συσσώρευσης (`rx, ry, gx, gy, bx, by`)
  είναι δηλωμένες **μέσα** στο εξωτερικό loop, οπότε είναι private ανά
  iteration/pixel και δεν χρειάζεται explicit `private()` clause.
- **Clamping inline αντί για συνάρτηση.** Στην GPU έκδοση το `clamp`
  είναι inlined με `if`-εκφράσεις, για να αποφύγω function calls μέσα στο
  hot loop.
- **Warm-up + 4 runs**, ίδιο μοτίβο με την Άσκηση 1.
- **Επαλήθευση pixel-pixel** ανάμεσα στις GPU εκδόσεις και τη σειριακή.

**Compile & run:**

```sh
clang -O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
      -DNUM_TEAMS=30 -DTHREAD_LIMIT=128 sobel.c -lm -o sobel
./sobel 1500.bmp
```

## Εκτέλεση όλων των πειραμάτων

Το `run_experiments.sh` τρέχει όλα τα σενάρια στο σύστημα **parallax** και
γράφει την έξοδο στο `results.txt`:

```sh
bash run_experiments.sh > results.txt 2>&1
```

Πριν την εκτέλεση το script κάνει `unset` σε `KMP_DEVICE_THREAD_LIMIT`,
`KMP_TEAMS_THREAD_LIMIT`, `OMP_THREAD_LIMIT` και θέτει `OMP_NUM_THREADS=1`
(η παράλληλη δουλειά γίνεται στη GPU, όχι στον host).

## Αναφορά

Όλες οι μετρήσεις, γραφικές παραστάσεις και η συζήτηση των αποτελεσμάτων
(speedup, επίδραση `num_teams`/`thread_limit`, σύγκριση combined vs nested)
βρίσκονται στο **`Anafora2.pdf`**.
