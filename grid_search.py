import ray
import shutil
import os

from ray import tune
from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper, MaximumIterationStopper, CombinedStopper
from ray.tune import Tuner, TuneConfig, RunConfig, CheckpointConfig
from ray.tune.execution.placement_groups import PlacementGroupFactory

from SLAC25.tune import Trainable
from SLAC25.utils import *
from SLAC25.network import *
from SLAC25.models import * # import the model

from search_space import search_space, resnet_sp

username = os.getlogin()
storage_path = f"/home/{username}/slac_experiments"
exp_name = f"BaselineCNN_lr"
exp_path = os.path.join(storage_path, exp_name)

os.makedirs(exp_path, mode=0o744, exist_ok=True)
print(f"Experimental files created under {exp_path}")

maxIterStopper = MaximumIterationStopper(100)

trailPlateauStoper = TrialPlateauStopper(
  metric="val_loss",
  std=0.001, # minimum improvement threshold
  num_results=10, # stop if 10 trials w/o any improvement consecutively
  grace_period=20, # min number of steps befor it can stop
  mode="min" 
)

combinedStopper = CombinedStopper(maxIterStopper, trailPlateauStoper)


ray.init()

trainable_with_resources = tune.with_resources(
    Trainable,
    resources=PlacementGroupFactory([{"CPU": 2, "GPU": 1}])
)

reporter = CLIReporter(
  metric_columns=["val_accuracy", "val_loss"]
)

tuner = Tuner(
  trainable_with_resources,
  param_space=search_space,
  tune_config=TuneConfig(
    metric="val_accuracy",
    mode="max",
    num_samples=5, # if grid_search, keep it =1 if wanna run all combination
    reuse_actors=False,
  ),
  run_config=RunConfig(
    name=exp_name,
    stop=combinedStopper,
    storage_path=exp_path,
    checkpoint_config=CheckpointConfig(
    # write ONLY at the very end
    checkpoint_frequency=0,
    checkpoint_at_end=True,
    num_to_keep=1,                       # keep one file max
    )
  ),
)

results = tuner.fit()
best_result = results.get_best_result("val_accuracy", mode="max")
best_config = best_result.config
print("Best config:", best_config)

save_dir = "~/best_model_checkpoint"


print("Files are in:", best_result.checkpoint.to_directory(save_dir))

ray.shutdown()