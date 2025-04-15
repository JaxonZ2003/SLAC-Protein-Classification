import os

for lr in [0.001, 0.01, 0.1]:
    out = f"/scratch/slac/slurmlogs/%j_lr_{lr}.out"
    err = f"/scratch/slac/slurmlogs/%j_lr_{lr}.err"
    odir = f"/scratch/slac/rcmodels/lr_{lr}"
    jobname = f"lr_{lr}"
    script = "/home/rebeccachang/capstone-SLAC/__main__.py"
    cmd = f'sbatch  -J {jobname} -N 1 --ntasks-per-node=1 -p gpu  -t 24:00:00  -o {out} -e {err} --gres=gpu:1 --wrap="python {script} --nepoch 100 --lr {lr} --batch_size=64 --verbose --maxImgs 10000 --outdir {odir}"'
    print(cmd)

    os.system(cmd)