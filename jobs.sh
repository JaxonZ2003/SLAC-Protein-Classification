#!/bin/bash
#SBATCH -J CNN_Resnet_20Epochs_0.01LR ## Job Name
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu ## Request 1 GPU   
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/slac/slurmlogs/outLog_%x_%j.txt ### Output Log File (Optional)
#SBATCH -e /scratch/slac/slurmlogs/errLog_%x_%j.txt ### Error Log File (Optional but suggest to have it)
#SBATCH -t 15:00:00 ### Job Execution Time

LR=0.001
srun -c 4 python ./__main__.py --num_workers 4 --nepoch 20 --lr $LR

  if [[ -f "$HOME/dials.sh" ]]; then
    source "$HOME/dials.sh"
  
  else
    echo "No dials.sh or modules found"
    exit 0
  fi
fi

echo "Environment set up okay"


LR=0.001
srun -c 2 python ./__main__.py --nepoch 20 --lr $LR
