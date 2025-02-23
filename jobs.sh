#!/bin/bash
#SBATCH -J CNN_Baseline_10Epochs ## Job Name
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu ## Request 1 GPU   
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/slac/slurmlogs/outLog_%x_%j.txt ### Output Log File (Optional)
#SBATCH -e /scratch/slac/slurmlogs/errLog_%x_%j.txt ### Error Log File (Optional but suggest to have it)
#SBATCH -t 10:00:00 ### Job Execution Time, 10 hour to see if it is running.

LR=0.001
srun -c 4 python ~/capstone-SLAC/Model_trainer.py --nepoch 10 --lr $LR


#srun -c 4 /home/reesekaro/train_wrapper.sh $LR

#cd $SLURM_SUBMIT_DIR

#/bin/hostname
#python ~/capstone-SLAC/Model_trainer.py