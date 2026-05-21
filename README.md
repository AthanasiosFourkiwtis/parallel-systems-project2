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

Ξεκίνησα από μια **naive έκδοση** όπου κάθε thread διάβαζε μόνο του από τη
global memory — ήταν πολύ αργή — και πέρασα σε **tiled υλοποίηση με shared
memory**. Χρησιμοποιώ δύο shared arrays: το `Bs` (`TILE_SIZE×TILE_SIZE`) και
το `As` (`BLOCK_ROWS×TILE_SIZE`, δηλαδή τόσο ψηλό όσο και το block), ώστε κάθε
στοιχείο του A/B να διαβάζεται από τη global memory **μία φορά ανά tile** και
να επαναχρησιμοποιείται από όλα τα threads του block.

Για το **block geometry** χρησιμοποιώ 2D block με σταθερό πλάτος:
`TILE_SIZE × BLOCK_ROWS`, όπου `BLOCK_ROWS = THREADS / TILE_SIZE`. Με
`TILE_SIZE=16` παίρνω 8/16/32 γραμμές για 128/256/512 threads/block
αντίστοιχα, οπότε το ίδιο kernel παραμετροποιείται μόνο από compile-time
flags (`-DN=...`, `-DTHREADS=...`).

Η **φόρτωση του B γίνεται με εσωτερικό loop**: επειδή το tile του B είναι
πάντα 16×16 ενώ το ύψος του block αλλάζει με τα threads, το `Bs` φορτώνεται
με loop `load_y += BLOCK_ROWS` ώστε να δουλεύει σωστά για όλες τις
διαμορφώσεις. Βάζω **`__syncthreads()` ανάμεσα στη φόρτωση και τον
υπολογισμό**, και ένα δεύτερο πριν το επόμενο tile, ώστε να μη χρησιμοποιεί
κανένα thread παλιά shared data ενώ άλλο γράφει νέα.

Φρόντισα η **χρονομέτρηση να είναι δίκαιη**: η σειριακή τρέχει 1 φορά (στο
N=2048 αργεί αρκετά), ενώ η GPU τρέχει 1 warm-up (δεν μετράει — αφορά
JIT/init κόστος) και 4 timed runs. Οι μεταφορές H2D και D2H **μετράνε** στον
GPU χρόνο, όπως ζητάει η εκφώνηση. Τέλος, κάνω **διπλή επαλήθευση**: το
αποτέλεσμα της GPU συγκρίνεται κατευθείαν με τον σειριακό υπολογισμό **και**
με το `Cmat<N>.txt`.

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

Έφτιαξα **δύο GPU εκδόσεις**. Η `sobel_omp_device` (**combined**) χρησιμοποιεί
μία οδηγία `target teams distribute parallel for collapse(2)`: τα δύο loops
(γραμμές × στήλες) ενώνονται σε έναν χώρο iterations και ο compiler μοιράζει
όλα τα pixels στα teams/threads. Η `sobel_omp_device_nested` (**nested**)
χρησιμοποιεί χωριστές οδηγίες: `target` → `teams distribute` (γραμμές στα
teams) → `parallel for` (στήλες στα threads).

Για το **mapping δεδομένων**, τα κανάλια εισόδου (`in_r/g/b`) και τα Sobel
kernels (`Gx`, `Gy`) μπαίνουν με `map(to:)` — μόνο διαβάζονται — ενώ τα
κανάλια εξόδου με `map(from:)`, αφού μόνο γράφονται και επιστρέφουν στον host.

Για την **επιλογή `num_teams` και `thread_limit`**, το `run_experiments.sh`
δοκιμάζει 15 ζευγάρια από `(32, 120)` μέχρι `(1024, 4)`, ώστε τα
`teams × thread_limit` να καλύπτουν τους 3840 cores της P40. Τα thread limits
είναι **πολλαπλάσια του 32**, επειδή το warp της NVIDIA έχει 32 threads —
διαφορετικά υπάρχουν idle lanes.

Όσον αφορά τα **race conditions**, οι μεταβλητές συσσώρευσης
(`rx, ry, gx, gy, bx, by`) είναι δηλωμένες **μέσα στο εσωτερικό loop** (στο
σώμα του `for j`, δηλαδή ανά pixel), οπότε είναι private ανά iteration/pixel
και δεν χρειάζεται explicit `private()` clause. Το **clamping το έκανα inline
αντί για συνάρτηση**: στην GPU έκδοση το `clamp` είναι inlined με
`if`-εκφράσεις, για να αποφύγω function calls μέσα στο hot loop.

Όπως και στην Άσκηση 1, χρησιμοποιώ **warm-up + 4 runs** και κάνω
**επαλήθευση pixel-pixel** ανάμεσα στις GPU εκδόσεις και τη σειριακή.

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
