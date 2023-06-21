#!/bin/bash

#SBATCH --job-name akas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=10G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -o slurm/logs/slurm-%A-%x.out

python ./train --cuda
python ./test --cuda

exit 0
