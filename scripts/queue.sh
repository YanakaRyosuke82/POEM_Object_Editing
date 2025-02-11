#!/bin/sh
#BSUB -q gpuh100
#BSUB -J marco
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o error/%J.out
#BSUB -e error/%J.err




bash scripts/interactive.sh BENCHMARK_VLM_QWEN_MATH_QWEN