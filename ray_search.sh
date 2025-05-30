#!/bin/bash
#SBATCH -J PLACEHOLDER
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o /home/jaxonz/logs/%x_%j.o
#SBATCH -e /home/jaxonz/logs/%x_%j.e
#SBATCH -t 150:00:00

if [[ -f "$HOME/dials.sh" ]]; then
  source "$HOME/dials.sh"
  
else
  echo "No dials.sh or modules found"
  exit 0
fi


echo "Environment set up okay"

python /home/jaxonz/SLAC/capstone-SLAC/ray_search.py -s 10 -r 50 -e 100
