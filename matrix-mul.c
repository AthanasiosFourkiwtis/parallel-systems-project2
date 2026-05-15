/*
 * MYE023 - Ergasia 2 - Askisi 1
 * Matmul me CUDA: C = A * B
 * Fourkiotis Athanasios, 4940
 *
 * compile:
 *   nvcc -O2 -x cu -DN=1024 -DTHREADS=128 matrix-mul.c -o matrix-mul
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>
/*Τα includes είναι για είσοδο/έξοδο, χρονομέτρηση και CUDA Runtime API. Το βασικό CUDA header είναι το cuda_runtime.h.*/

#ifndef N
#define N 1024
#endif

#ifndef THREADS
#define THREADS 128
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
/*Τα N, THREADS και TILE_SIZE είναι compile-time παράμετροι. Με το script κάνω compile το ίδιο πρόγραμμα για διαφορετικά
 μεγέθη πινάκων και διαφορετικό πλήθος threads ανά block.*/

#if (THREADS % TILE_SIZE) != 0
#error "THREADS must be divisible by TILE_SIZE"
#endif

#define BLOCK_ROWS (THREADS / TILE_SIZE)
/*Θέλω τα threads να διαιρούνται με το tile size, γιατί φτιάχνω 2D block με πλάτος 16 και ύψος THREADS/16.*/

/* Τους πίνακες τους δήλωσα global static γιατί για μεγάλα N, ειδικά N=2048, είναι μεγάλοι και δεν θέλω να δεσμευτούν στο stack. */
static int A[N][N];
static int B[N][N];
static int C_gpu[N][N];   /*αποτελεσμα GPU*/
static int C_serial[N][N]; /*αποτελεσμα CPU*/

static char Afile[32], Bfile[32], Cfile[32];

/* Η check_cuda ελέγχει αν μία CUDA κλήση πέτυχε. Αν υπάρχει σφάλμα, τυπώνει μήνυμα και σταματάει το πρόγραμμα. Είναι χρήσιμο για να μην συνεχίσω με λάθος δεδομένα. */
static void check_cuda(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

/* Η readmat διαβάζει έναν NxN πίνακα από αρχείο και τον αποθηκεύει γραμμικά. Το στοιχείο [i][j] μπαίνει στη θέση i*n+j. */
int readmat(const char *fname, int *mat, int n)
{
    FILE *fp;
    int i, j;

    fp = fopen(fname, "r");
    if (fp == NULL)
        return -1;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            if (fscanf(fp, "%d", &mat[i*n+j]) != 1) {
                fclose(fp);
                return -1;
            }

    fclose(fp);
    return 0;
}

/* seiriako matmul gia epalitheysi (parmeno apo to matmul_serial.c) */
void matmul_serial_host(int *A, int *B, int *C, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i*n + k] * B[k*n + j];
            C[i*n + j] = sum;
        }
}

/* tiled kernel με shared memory.
 * Eixa ftiaksi proti tin naive ekdosi (kathe thread diavazei mono tou apo
 * global) alla itan polly pio argi, etsi pirage stin tiled. */
__global__ void matmul(int *A, int *B, int *C, int n) /*Αυτό είναι το kernel που τρέχει στη GPU. Παίρνει pointers προς τους πίνακες που βρίσκονται στη GPU.*/
{
    __shared__ int As[BLOCK_ROWS][TILE_SIZE]; /*Δηλώνω δύο shared arrays, ένα για κομμάτι του A και ένα για κομμάτι του B. Αυτά τα χρησιμοποιούν τα threads του ίδιου block.*/
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  /*Με τα blockIdx και threadIdx βρίσκω το row και col του C που αντιστοιχεί στο συγκεκριμένο thread.*/
    int sum = 0;

    for (int tile = 0; tile < n; tile += TILE_SIZE) {  /*Το loop περνάει τη διάσταση k σε tiles των 16 στοιχείων.*/
        int a_col = tile + threadIdx.x;      /*Κάθε thread φορτώνει ένα στοιχείο του A στη shared memory. Οι έλεγχοι ορίων αποφεύγουν παράνομες προσβάσεις στη μνήμη.*/
        if (row < n && a_col < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        for (int load_y = threadIdx.y; load_y < TILE_SIZE; load_y += BLOCK_ROWS) { /*Το loop για το B υπάρχει επειδή το tile του B είναι 16×16, ενώ το ύψος
													του block αλλάζει ανάλογα με τα threads. Έτσι η φόρτωση δουλεύει σωστά και για 128, και για 256, και για 512 threads.*/
            int b_row = tile + load_y;
            if (b_row < n && col < n)
                Bs[load_y][threadIdx.x] = B[b_row * n + col];
            else
                Bs[load_y][threadIdx.x] = 0;
        }

        __syncthreads(); /*Περιμένω όλα τα threads του block να τελειώσουν τη φόρτωση στη shared memory πριν αρχίσει ο υπολογισμός.*/

        for (int k = 0; k < TILE_SIZE; k++)   /*Για το κάθε tile, το thread πολλαπλασιάζει μία γραμμή από το As με μία στήλη από το Bs και προσθέτει στο sum.*/
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads(); /*Το δεύτερο __syncthreads() εξασφαλίζει ότι κανένα thread δεν χρησιμοποιεί ακόμα το παλιό tile πριν γραφτεί το επόμενο στη shared memory.*/
    }

    if (row < n && col < n)
        C[row * n + col] = sum;  /*Στο τέλος κάθε thread γράφει το τελικό αποτέλεσμα στη θέση C[row*n+col], αν η θέση είναι εντός ορίων.*/
}

/* sygkrisi GPU vs seiriako apotelesma */
int verify_gpu_vs_serial(void)
{
    int errors = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (C_gpu[i][j] != C_serial[i][j]) {
                errors++;
                if (errors <= 10)
                    printf("Mismatch at C[%d][%d]: gpu=%d, serial=%d\n",
                           i, j, C_gpu[i][j], C_serial[i][j]);
            }
    return errors;
}

/* sygkrisi GPU vs to etoimo Cmat<N>.txt */
int verify_gpu_vs_file(const char *fname)
{
    FILE *fp = fopen(fname, "r");
    int errors = 0;

    if (fp == NULL) {
        fprintf(stderr, "Warning: cannot open %s for expected-result verification\n", fname);
        return -1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int expected;
            if (fscanf(fp, "%d", &expected) != 1) {
                fprintf(stderr, "Warning: invalid or incomplete expected-result file %s\n", fname);
                fclose(fp);
                return -1;
            }
            if (C_gpu[i][j] != expected) {
                errors++;
                if (errors <= 10)
                    printf("Mismatch vs %s at C[%d][%d]: gpu=%d, expected=%d\n",
                           fname, i, j, C_gpu[i][j], expected);
            }
        }
    }

    fclose(fp);
    return errors;
}

double get_time(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

int main(void)
{
    int *d_A, *d_B, *d_C; /*Οι pointers με d_ δείχνουν σε μνήμη της GPU*/
    int size = N * N * sizeof(int); /*Το size είναι το μέγεθος ενός NxN πίνακα σε bytes και το χρειάζομαι για cudaMalloc και cudaMemcpy.*/
    dim3 threads_per_block(TILE_SIZE, BLOCK_ROWS); /*Χρησιμοποιώ 2D blocks με πλάτος 16 και ύψος BLOCK_ROWS, ώστε να έχω συνολικά 128, 256 ή 512 threads ανά block.*/
    dim3 blocks_per_grid((N + TILE_SIZE - 1) / TILE_SIZE,
                         (N + BLOCK_ROWS - 1) / BLOCK_ROWS); /*Υπολογίζω αυτόματα πόσα blocks χρειάζονται σε x και y, ώστε να καλυφθούν 
						 όλες οι γραμμές και στήλες του πίνακα C.*/
    int num_blocks = blocks_per_grid.x * blocks_per_grid.y;
    int runs = 4;
    double t1, t2;

    snprintf(Afile, sizeof(Afile), "Amat%d.txt", N);
    snprintf(Bfile, sizeof(Bfile), "Bmat%d.txt", N);
    snprintf(Cfile, sizeof(Cfile), "Cmat%d.txt", N);

    printf("Matrix multiplication (CUDA + Serial baseline)\n");
    printf("N=%d, threads/block=%d (%dx%d), blocks=%d (%dx%d), tile=%d\n",
           N, THREADS, TILE_SIZE, BLOCK_ROWS, num_blocks,
           blocks_per_grid.x, blocks_per_grid.y, TILE_SIZE);
    printf("Files: %s, %s, expected %s\n\n", Afile, Bfile, Cfile);

    /* to I/O den metraei sti xronometrisi */
    if (readmat(Afile, (int *)A, N) < 0) {
        fprintf(stderr, "Error: cannot open %s\n", Afile);
        return 1;
    }
    if (readmat(Bfile, (int *)B, N) < 0) {
        fprintf(stderr, "Error: cannot open %s\n", Bfile);
        return 1;
    }

    /* trexoume to seiriako mia fora (sto N=2048 paira pollh wra) */
    printf("--- SERIAL (host) ---\n");
    t1 = get_time();
    matmul_serial_host((int *)A, (int *)B, (int *)C_serial, N);
    t2 = get_time();
    double serial_time = t2 - t1;
    printf("Serial: %.6f sec\n\n", serial_time);

    /* sti xronometrisi tis GPU vazoume kai tis metafores host<->device */
    check_cuda(cudaMalloc((void **)&d_A, size), "cudaMalloc d_A");
    check_cuda(cudaMalloc((void **)&d_B, size), "cudaMalloc d_B");
    check_cuda(cudaMalloc((void **)&d_C, size), "cudaMalloc d_C");

    /* warm-up gia na min metrhthei to JIT/init kostos sthn 1h ektelhsh */
    printf("--- GPU (CUDA) - warm-up + 4 timed runs ---\n");
    check_cuda(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice), "warm-up cudaMemcpy H2D A");
    check_cuda(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice), "warm-up cudaMemcpy H2D B");
    matmul<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);
    check_cuda(cudaGetLastError(), "warm-up kernel launch");
    check_cuda(cudaDeviceSynchronize(), "warm-up cudaDeviceSynchronize");
    check_cuda(cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost), "warm-up cudaMemcpy D2H C");
    printf("GPU warm-up: done (not counted)\n");

    double total_gpu = 0.0;
    for (int r = 0; r < runs; r++) {
        t1 = get_time();

        check_cuda(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D A");
        check_cuda(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D B");

        matmul<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);

        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        check_cuda(cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H C");

        t2 = get_time();
        double t = t2 - t1;
        printf("GPU run %d: %.6f sec\n", r + 1, t);
        total_gpu += t;
    }
    double gpu_avg = total_gpu / runs;
    printf("GPU avg: %.6f sec\n\n", gpu_avg);

    /* dipli epalitheysi */
    int errors_serial = verify_gpu_vs_serial();
    int errors_file = verify_gpu_vs_file(Cfile);

    printf("--- RESULTS ---\n");
    printf("  Serial:        %.6f sec\n", serial_time);
    printf("  GPU avg:       %.6f sec\n", gpu_avg);
    printf("  Speedup:       %.2fx\n", serial_time / gpu_avg);
    printf("  Verification vs serial: %s\n", errors_serial == 0 ? "OK" : "WRONG");
    if (errors_serial > 0)
        printf("  (%d mismatched elements vs serial)\n", errors_serial);
    printf("  Verification vs %s: %s\n", Cfile,
           errors_file == 0 ? "OK" : (errors_file < 0 ? "SKIPPED" : "WRONG"));
    if (errors_file > 0)
        printf("  (%d mismatched elements vs %s)\n", errors_file, Cfile);

    /* free apo GPU */
    check_cuda(cudaFree(d_A), "cudaFree d_A");
    check_cuda(cudaFree(d_B), "cudaFree d_B");
    check_cuda(cudaFree(d_C), "cudaFree d_C");

    return 0;
}
