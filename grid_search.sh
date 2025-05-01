#!/bin/bash
#SBATCH -J BaselineGridSearch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=$HOME/logs/%x_%j.out
#SBATCH --error=$HOME/logs/%x_%j.err
#SBATCH -t 2:00:00

# source ~/dials.sh

python ./grid_search.py
