#!/bin/bash
# MYE023 Assignment 2 - script that runs all the experiments on parallax
# Athanasios Fourkiotis 4940
#
# usage:  bash run_experiments.sh > results.txt 2>&1

set -e

# set host threads = 1 so libomp does not spam warnings
# (the parallel work happens on the GPU via target offloading)
export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=2

# some env-vars (KMP/OMP thread limits) trigger the annoying
# "Cannot form a team with 120 threads" - unset them
unset KMP_DEVICE_THREAD_LIMIT
unset KMP_TEAMS_THREAD_LIMIT
unset OMP_THREAD_LIMIT

echo "=========================================="
echo "EXERCISE 1 - CUDA matmul"
echo "=========================================="

# every (N, THREADS) pair the handout asks for
for N in 512 1024 2048; do
    for T in 128 256 512; do
        echo ""
        echo "----- N=$N THREADS=$T -----"
        nvcc -O2 -x cu -DN=$N -DTHREADS=$T matrix-mul.c -o matrix-mul
        ./matrix-mul
    done
done

echo ""
echo "=========================================="
echo "EXERCISE 2 - Sobel"
echo "=========================================="

# (threads_per_team, num_teams) chosen to cover the 3840 cores of the P40
# plus two extras (512x8, 1024x4 = 4096) for comparison
CONFIGS=(
    "32 120"
    "64 60"
    "96 40"
    "128 30"
    "160 24"
    "192 20"
    "256 15"
    "320 12"
    "384 10"
    "480 8"
    "512 8"
    "640 6"
    "768 5"
    "960 4"
    "1024 4"
)

for IMG in 500.bmp 1000.bmp 1500.bmp; do
    echo ""
    echo "=========================================="
    echo "SOBEL IMAGE=$IMG"
    echo "=========================================="

    for cfg in "${CONFIGS[@]}"; do
        set -- $cfg
        THREADS=$1
        TEAMS=$2
        echo ""
        echo "----- IMAGE=$IMG THREAD_LIMIT=$THREADS NUM_TEAMS=$TEAMS -----"
        clang -O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
              -DNUM_TEAMS=$TEAMS -DTHREAD_LIMIT=$THREADS \
              sobel.c -lm -o sobel
        ./sobel "$IMG"
    done
done

echo ""
echo "=========================================="
echo "end of experiments"
echo "=========================================="
