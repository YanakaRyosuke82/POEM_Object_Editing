#!/bin/sh
#BSUB -q gpuh100
#BSUB -J marco
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o error/marco_TEST_WITHTHHEIRCOMMAND.out
#BSUB -e error/marco_TEST_WITHTHHEIRCOMMAND.err



# bash scripts/run_interactive.sh benchmark
bash scripts/interactive.sh benchmark