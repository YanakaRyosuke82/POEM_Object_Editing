#!/bin/sh
#BSUB -q gpuh100
#BSUB -J marco
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o error/MARCO_BENCHMARK.out
#BSUB -e error/MARCO_BENCHMARK.err




bash scripts/interactive.sh MARCO_BENCHMARK