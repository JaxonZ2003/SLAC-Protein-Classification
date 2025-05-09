#!/bin/bash
#SBATCH -J BaseCNNstf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o /home/jaxonz/logs/%x_%j.o
#SBATCH -e /home/jaxonz/logs/%x_%j.e
#SBATCH -t 48:00:00

# source ~/dials.sh

if [[ -f "$HOME/dials.sh" ]]; then
  source "$HOME/dials.sh"
  
else
  echo "No dials.sh or modules found"
  exit 0
fi


echo "Environment set up okay"

python /home/jaxonz/SLAC/capstone-SLAC/grid_search.py
