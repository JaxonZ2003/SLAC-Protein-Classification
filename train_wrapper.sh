#!/bin/bash
# meant to be run inside an srun command

hostname
ME=$(whoami)
LR=$1
OUTDIR=/scratch/slac/models/${ME}.${SLURM_JOB_NAME}.${SLURM_JOB_ID}
echo $OUTDIR
python ~/capstone-SLAC/Model_trainer.py --outdir $OUTDIR --lr $LR 10
