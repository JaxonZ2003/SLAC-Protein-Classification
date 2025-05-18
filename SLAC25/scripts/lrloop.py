import os
import sys

tag = sys.argv[1]

queue="regular"
for i, lr in enumerate([0.001, 0.01, 0.1]):
    out = f"/pscratch/sd/r/rebecca/slurmlogs/{tag}_%j_lr_{lr}.out"
    err = f"/pscratch/sd/r/rebecca/slurmlogs/{tag}_%j_lr_{lr}.err"
    odir = f"/pscratch/sd/r/rebecca/rcmodels/{tag}_lr_{lr}"
    jobname = f"{tag}.{i}"
    script = "/global/homes/r/rebecca/capstone-SLAC/__main__.py"
    cmd = f'sbatch  -J {jobname} -N 1 --ntasks-per-node=1 -A m4731_g -t 24:00:00  -o {out} -e {err} --gres=gpu:1 --wrap="python {script} --nepoch 100 --lr {lr} --batch_size=64 --verbose --maxImgs 10000 --nwork 0 --outdir {odir}" -q {queue} -C gpu'
    print(cmd)

    os.system(cmd)
