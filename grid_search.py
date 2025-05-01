import ray
import shutil

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper, MaximumIterationStopper, CombinedStopper
from ray.tune import Tuner, TuneConfig, RunConfig, CheckpointConfig
from ray.tune.execution.placement_groups import PlacementGroupFactory


from SLAC25.utils import *
from SLAC25.network import Trainable
from SLAC25.models import * # import the model

maxIterStopper = MaximumIterationStopper(10)

trailPlateauStoper = TrialPlateauStopper(
  metric="val_loss",
  std=0.001, # minimum improvement threshold
  num_results=3, # stop if 3 trials w/o any improvement consecutively
  grace_period=5, # min number of steps befor it can stop
  mode="min" 
)

combinedStopper = CombinedStopper(maxIterStopper, trailPlateauStoper)

search_space = {
  "model": "BaselineCNN",
  "num_classes": 4,
  "keep_prob": tune.grid_search([0.75]),
  "num_epochs": -1,
  "lr": tune.grid_search([0.001, 0.01]),
  "batch_size": 32,
  "verbose": False,
  "testmode": True,
  "outdir": "./models",
  "tune": False,
  "seed": 1
}

ray.init()

trainable_with_resources = tune.with_resources(
    Trainable,
    resources=PlacementGroupFactory([{"CPU": 2, "GPU": 1}])
)

tuner = Tuner(
  trainable_with_resources,
  param_space=search_space,
  tune_config=TuneConfig(
    metric="val_accuracy",
    mode="max",
    num_samples=1, # if grid_search, keep it one if wanna run all combination
    reuse_actors=False,
  ),
  run_config=RunConfig(
    stop=combinedStopper,
    storage_path="~/ray_out",
    checkpoint_config=CheckpointConfig(
    #  write ONLY at the *very end*
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