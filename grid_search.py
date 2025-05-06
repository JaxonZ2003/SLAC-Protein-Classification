import ray
import shutil

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper, MaximumIterationStopper, CombinedStopper
from ray.tune import Tuner, TuneConfig, RunConfig, CheckpointConfig
from ray.tune.execution.placement_groups import PlacementGroupFactory


from SLAC25.utils import *
from SLAC25.network import Trainable
from SLAC25.models import * # import the model

search_space = {
  "model": "BaselineCNN",
  "num_classes": 4,
  "keep_prob": tune.grid_search([0.75]),
  "num_epochs": -1,
  "lr": tune.grid_search([0.001, 0.01, 0.1]),
  "batch_size": 32,
  "verbose": False,
  "testmode": False,
  "outdir": "./models",
  "tune": True,
  "seed": 1
}

resnet_sp = {
  "model": "ResNet",
  "num_classes": 4,
  "num_epochs": 10,
  "batch_size": 32,
  "verbose": True,
  "testmode": False,
  "outdir": "./models",
  "tune": True,
  "seed": 1,
  "hidden_dim": tune.choice([256, 512]),
  "lr": tune.loguniform(1e-4, 1e-1),
  "keep_prob": tune.uniform(0.5, 0.9),
  "beta1": tune.uniform(0.5, 0.9),
  "beta2": tune.uniform(0.5, 0.9),
}

maxIterStopper = MaximumIterationStopper(10)

trailPlateauStoper = TrialPlateauStopper(
  metric="val_loss",
  std=0.001, # minimum improvement threshold
  num_results=3, # stop if 3 trials w/o any improvement consecutively
  grace_period=5, # min number of steps befor it can stop
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
  param_space=resnet_sp,
  tune_config=TuneConfig(
    metric="val_accuracy",
    mode="max",
    num_samples=5, # if grid_search, keep it =1 if wanna run all combination
    reuse_actors=False,
  ),
  run_config=RunConfig(
    name="ResNet_Tune",
    stop=combinedStopper,
    storage_path="~/ray_out",
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