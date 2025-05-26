#!/bin/bash
#SBATCH -J BestR3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o /home/jaxonz/logs/%x_%j.o
#SBATCH -e /home/jaxonz/logs/%x_%j.e
#SBATCH -t 150:00:00

# source ~/dials.sh

if [[ -f "$HOME/dials.sh" ]]; then
  source "$HOME/dials.sh"
  
else
  echo "No dials.sh or modules found"
  exit 0
fi


echo "Environment set up okay"

python /home/jaxonz/SLAC/capstone-SLAC/grid_search.py -s 10 -r 50 -e 100
# python /home/jaxonz/SLAC/capstone-SLAC/rough_script.py
