#!/bin/sh
#BSUB -q gpuh100
#BSUB -J marco
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o marco%J.out
#BSUB -e marco%J.err
##BSUB -u marscho@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 




# # Loop to run jobs for dataset identifiers
# for i in $(seq 1 10)
# do
#     bash train_marco_interactive.sh "marco" "exp$i"
# done

 bash train_marco_interactive.sh "VAE"