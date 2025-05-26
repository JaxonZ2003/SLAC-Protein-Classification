import ray
import shutil
import os
import getpass

from argparse import ArgumentParser
from ray import tune
from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper, MaximumIterationStopper, CombinedStopper
from ray.tune import Tuner, TuneConfig, RunConfig, CheckpointConfig, FailureConfig
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.schedulers import ASHAScheduler

from SLAC25.tune import Trainable
from SLAC25.utils import *
from SLAC25.network import *
from SLAC25.models import * # import the model

from search_space import Experiments

def run_tune(storage_path, exp_name, search_space, max_epoch=10, grace=5, patience=3, delta=0.001, grid_search=False, scheduler_name=None):
  '''
  Tune for the hyperparameters for at most `max_epoch`.
  Stop if
  - Reaching `max_epoch` epochs per trial
  - No improvement >= `delta` for `patience` epochs consecutively
  '''

  exp_path = os.path.join(storage_path, exp_name)

  num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
  num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE") or
                  os.environ.get("SLURM_GPUS_PER_NODE") or
                  len(os.environ.get("SLURM_JOB_GPUS","").split(",")) or 0)

  ray.init(
      num_cpus=num_cpus,
      num_gpus=num_gpus,
      object_store_memory=4 * 1024**3      # 4 GB for the plasma store
  )

  # ray.init(address="auto")

  trainable_with_resources = tune.with_resources(
      Trainable,
      resources=PlacementGroupFactory([{"CPU": 1, "GPU": 1}])
  )

  reporter = CLIReporter(
    metric_columns=["train_accuracy", "train_loss", "val_accuracy", "val_loss", "test_accuracy", "test_loss"],
    max_report_frequency=1800 # print to the terminal every 1800s
  )


  os.makedirs(storage_path, mode=0o744, exist_ok=True)
  print("Experiment Setup")
  print(f"• Experiment Name : {exp_name}")
  print(f"• Storage Path    : {exp_path}")
  print(f"• Grid Search     : {grid_search}")
  print(f"• Scheduler       : {scheduler_name}")
  print(f"• Metric          : Validation Loss")
  print("="*40)

  if scheduler_name == "ASHA":
    scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='val_accuracy',
    mode='max',
    max_t=max_epoch,                       # largest running epochs
    grace_period=5,
    reduction_factor=3,             # keeps 1/3 of the trials eventually
    brackets=1)

    print("Scheduler Configuration")
    print(f"• Scheduler Name                 : {scheduler_name + "Scheduler"}")
    print(f"• Minimum Epochs (Grace Period)  : {grace}")
    print(f"• Reduction Factor               : 3")
    print(f"• Maximum Running Epochs         : {max_epoch}")
    print("="*40)

    tuner = Tuner(
      trainable_with_resources,
      param_space=search_space,
      tune_config=TuneConfig(
      num_samples=10, # if grid_search, keep it =1 if wanna run all combination
      scheduler=scheduler,
      reuse_actors=False,
      max_concurrent_trials=2
    ),run_config=RunConfig(
      name=exp_name,
      # resources_per_trial={"cpu": 2, "gpu": 1},
      # stop=combinedStopper,
      storage_path=storage_path,
      progress_reporter=reporter,
      failure_config=FailureConfig(max_failures=3, fail_fast=False),
      checkpoint_config=CheckpointConfig(
      # write ONLY at the very end
      checkpoint_frequency=1,
      checkpoint_at_end=True,
      num_to_keep=1,                       # keep one file max
      )
    )
  )
  else:
    maxIterStopper = MaximumIterationStopper(max_epoch)

    trailPlateauStoper = TrialPlateauStopper(
    metric="val_loss",
    std=delta, # minimum improvement threshold
    num_results=patience, # stop if # trials w/o any improvement consecutively
    grace_period=grace, # min number of steps before it can stop
    mode="min" 
    )

    combinedStopper = CombinedStopper(maxIterStopper, trailPlateauStoper)
    print("Early Stopper Configuration")
    print(f"• Minimum Epochs (Grace Period)  : {grace}")
    print(f"• Patience (No Improvement Limit): {patience}")
    print(f"• Improvement Threshold (min Δ)  : {delta}")
    print("="*40)


    tuner = Tuner(
      trainable_with_resources,
      param_space=search_space,
      tune_config=TuneConfig(
        metric="val_accuracy" if not search_space.get("lr_scheduler", False) else "test_accuracy",
        mode="max",
        num_samples=5 if not grid_search else 1, # if grid_search, keep it =1 if wanna run all combination
        reuse_actors=False,
        max_concurrent_trials=2
      ),
      run_config=RunConfig(
        name=exp_name,
        stop=combinedStopper,
        storage_path=storage_path,
        progress_reporter=reporter,
        failure_config=FailureConfig(max_failures=3, fail_fast=False),
        checkpoint_config=CheckpointConfig(
        # write ONLY at the very end
        checkpoint_frequency=2,
        checkpoint_at_end=True,
        num_to_keep=1,                       # keep one file max
        )
      ),
    )
  
  print(f"Initialization complete")

  print(f"Start tuning...")
  results = tuner.fit()
  print(f"Tuning complete")

  if results.num_completed_trials:
      best = results.get_best_result(metric="val_accuracy", mode="max")
      ckpt_dir = best.checkpoint.to_directory(os.path.expanduser(
                    f"~/best_model_checkpoint/{exp_name}"))
      print("Best checkpoint saved to:", ckpt_dir)
  else:
      print("No trials finished successfully. See", results.logdir)

  ray.shutdown()


if __name__ == "__main__":
  username = getpass.getuser()
  storage_path = f"/home/{username}/slac_experiments"
  exp_name = f"BaselineCNN_lr"

  ap = ArgumentParser()
  ap.add_argument('-p', '--storage', type=str, default=storage_path, help="Main directory where the experiments stored")
  ap.add_argument('-x', '--experiment', type=str, default="Unnamed", help="Sub-directory where individual experiment stored")
  ap.add_argument('-s', '--searchspace', type=str, default=1, help="Hyperparameters searching space")
  ap.add_argument('-e', '--epoch', type=int, default=20, help="Maximum number of epochs a trial can run")
  ap.add_argument('-r', '--grace', type=int, default=5, help="Minimum number of epochs before a stopper can intervene")
  ap.add_argument('-n', '--patience', type=int, default=3, help="Maximum number of epochs allowing no improvement without stopping")
  ap.add_argument('-d', '--delta', type=float, default=0.001, help="Threshold that a minimum improvement should reach")
  ap.add_argument('--scheduler', '--sche', type=str, default=None, help="Scheduler used during tuning")
  ap.add_argument('-g', '--grid', action='store_true', help="Turn on the grid search")
  args = ap.parse_args()

  search_space = Experiments[args.searchspace]
  exp_name = search_space.get("name", "Unnamed")

  run_tune(storage_path=args.storage,
           exp_name=exp_name,
           search_space=search_space,
           max_epoch=args.epoch,
           grace=args.grace,
           patience=args.patience,
           delta=args.delta,
           grid_search=args.grid,
           scheduler_name=args.scheduler)