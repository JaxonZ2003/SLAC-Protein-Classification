#!/bin/bash
#SBATCH -J gsBsCNN
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o /home/jaxonz/logs/%x_%j.o
#SBATCH -e /home/jaxonz/logs/%x_%j.e
#SBATCH -t 24:00:00

# source ~/dials.sh

python /home/jaxonz/SLAC/capstone-SLAC/grid_search.py
