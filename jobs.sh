#!/bin/bash
#SBATCH -J CNN_Resnet_20Epochs_0.01LR ## Job Name
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o ~/logs/outLog_%x_%j.txt
#SBATCH -e ~/logs/errLog_%x_%j.txt
#SBATCH -t 1:00:00 ### Job Execution Time

if [[ -f "$HOME/dials.sh" ]]; then
  source "$HOME/dials.sh"
  
else
  echo "No dials.sh or modules found"
  exit 0
fi


echo "Environment set up okay"


LR=0.001
python ./__main__.py --nepoch 5 --lr 0.001
