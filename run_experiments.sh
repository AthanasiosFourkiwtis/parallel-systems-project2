#!/bin/bash
# MYE023 Ergasia 2 - script gia trexei ola ta peiramata sto parallax
# Athanasios Fourkiotis 4940
#
# treximo:  bash run_experiments.sh > results.txt 2>&1

set -e

# kanw host threads = 1 wste na min spamarei to libomp warnings
# (i parallili douleia ginete sti GPU mesw target offloading)
export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=2

# kapoia env-vars (KMP/OMP thread limits) prokaloun to enoxlitiko
# "Cannot form a team with 120 threads" - ta vazoume unset
unset KMP_DEVICE_THREAD_LIMIT
unset KMP_TEAMS_THREAD_LIMIT
unset OMP_THREAD_LIMIT

echo "=========================================="
echo "ASKISI 1 - CUDA matmul"
echo "=========================================="

# ola ta zeugaria (N, THREADS) pou zita i ekfwnisi
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
echo "ASKISI 2 - Sobel"
echo "=========================================="

# (threads_per_team, num_teams) wste na piasoume ta 3840 cores tis P40
# vazw kai dyo eytra (512x8, 1024x4 = 4096) gia sygkrisi
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
echo "telos peiramatwn"
echo "=========================================="
