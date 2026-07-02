/*
 * MYE023 - Assignment 2 - Exercise 1
 * Matmul with CUDA: C = A * B
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
/* The includes cover I/O, timing, and the CUDA Runtime API. The core CUDA header is cuda_runtime.h. */

#ifndef N
#define N 1024
#endif

#ifndef THREADS
#define THREADS 128
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
/* N, THREADS, and TILE_SIZE are compile-time parameters. The script compiles the same program
 for different matrix sizes and different thread counts per block. */

#if (THREADS % TILE_SIZE) != 0
#error "THREADS must be divisible by TILE_SIZE"
#endif

#define BLOCK_ROWS (THREADS / TILE_SIZE)
/* THREADS must be divisible by the tile size, since the 2D block is 16 wide and THREADS/16 tall. */

/* The matrices are declared global static because for large N, especially N=2048, they are big and should not live on the stack. */
static int A[N][N];
static int B[N][N];
static int C_gpu[N][N];   /* GPU result */
static int C_serial[N][N]; /* CPU result */

static char Afile[32], Bfile[32], Cfile[32];

/* check_cuda verifies that a CUDA call succeeded. On error it prints a message and aborts, so we never continue with bad data. */
static void check_cuda(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

/* readmat reads an NxN matrix from a file and stores it linearly. Element [i][j] goes to position i*n+j. */
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

/* serial matmul for verification (taken from matmul_serial.c) */
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

/* tiled kernel with shared memory.
 * I first wrote the naive version (each thread reading on its own from
 * global memory) but it was far slower, so I moved to the tiled one. */
__global__ void matmul(int *A, int *B, int *C, int n) /* This is the kernel that runs on the GPU. It takes pointers to the matrices living in GPU memory. */
{
    __shared__ int As[BLOCK_ROWS][TILE_SIZE]; /* Two shared arrays, one for a chunk of A and one for a chunk of B. They are shared by the threads of the same block. */
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  /* blockIdx and threadIdx give the row and col of C this particular thread is responsible for. */
    int sum = 0;

    for (int tile = 0; tile < n; tile += TILE_SIZE) {  /* The loop walks the k dimension in tiles of 16 elements. */
        int a_col = tile + threadIdx.x;      /* Each thread loads one element of A into shared memory. The bounds checks prevent illegal memory accesses. */
        if (row < n && a_col < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        for (int load_y = threadIdx.y; load_y < TILE_SIZE; load_y += BLOCK_ROWS) { /* The loop over B exists because B's tile is 16×16 while the block height
													varies with the thread count. This way the load works correctly for 128, 256, and 512 threads alike. */
            int b_row = tile + load_y;
            if (b_row < n && col < n)
                Bs[load_y][threadIdx.x] = B[b_row * n + col];
            else
                Bs[load_y][threadIdx.x] = 0;
        }

        __syncthreads(); /* Wait for every thread in the block to finish loading into shared memory before the computation starts. */

        for (int k = 0; k < TILE_SIZE; k++)   /* For each tile, the thread multiplies one row of As by one column of Bs and accumulates into sum. */
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads(); /* The second __syncthreads() guarantees no thread is still using the old tile before the next one is written into shared memory. */
    }

    if (row < n && col < n)
        C[row * n + col] = sum;  /* Finally each thread writes its result to C[row*n+col], provided the position is in bounds. */
}

/* compare the GPU result against the serial one */
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

/* compare the GPU result against the provided Cmat<N>.txt */
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
    int *d_A, *d_B, *d_C; /* the d_ pointers point to GPU memory */
    int size = N * N * sizeof(int); /* size is the byte size of one NxN matrix, needed for cudaMalloc and cudaMemcpy. */
    dim3 threads_per_block(TILE_SIZE, BLOCK_ROWS); /* 2D blocks, 16 wide and BLOCK_ROWS tall, giving 128, 256, or 512 threads per block in total. */
    dim3 blocks_per_grid((N + TILE_SIZE - 1) / TILE_SIZE,
                         (N + BLOCK_ROWS - 1) / BLOCK_ROWS); /* automatically compute how many blocks are needed in x and y
						 to cover every row and column of C. */
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

    /* I/O is not part of the timed section */
    if (readmat(Afile, (int *)A, N) < 0) {
        fprintf(stderr, "Error: cannot open %s\n", Afile);
        return 1;
    }
    if (readmat(Bfile, (int *)B, N) < 0) {
        fprintf(stderr, "Error: cannot open %s\n", Bfile);
        return 1;
    }

    /* run the serial version once (at N=2048 it takes very long) */
    printf("--- SERIAL (host) ---\n");
    t1 = get_time();
    matmul_serial_host((int *)A, (int *)B, (int *)C_serial, N);
    t2 = get_time();
    double serial_time = t2 - t1;
    printf("Serial: %.6f sec\n\n", serial_time);

    /* GPU timing includes the host<->device transfers */
    check_cuda(cudaMalloc((void **)&d_A, size), "cudaMalloc d_A");
    check_cuda(cudaMalloc((void **)&d_B, size), "cudaMalloc d_B");
    check_cuda(cudaMalloc((void **)&d_C, size), "cudaMalloc d_C");

    /* warm-up, so the JIT/init cost of the 1st launch is not measured */
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

    /* double verification */
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

    /* free GPU memory */
    check_cuda(cudaFree(d_A), "cudaFree d_A");
    check_cuda(cudaFree(d_B), "cudaFree d_B");
    check_cuda(cudaFree(d_C), "cudaFree d_C");

    return 0;
}
