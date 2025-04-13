import ray

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune import Tuner, TuneConfig, RunConfig
from ray.tune.execution.placement_groups import PlacementGroupFactory


from SLAC25.utils import *
from SLAC25.network import Trainable
from SLAC25.models import * # import the model

stopper = TrialPlateauStopper(
  metric="val_loss",
  std=0.001, # minimum improvement threshold
  num_results=3, # stop if 3 trials w/o any improvement consecutively
  grace_period=5, # min number of steps befor it can stop
  mode="min" 
)

search_space = {
  "model": "BaselineCNN",
  "num_classes": 4,
  "keep_prob": tune.choice([0.75]),
  "num_epochs": tune.choice([3]),
  "lr": tune.choice([0.001, 0.01]),
  "batch_size": 32,
  "verbose": True,
  "testmode": True,
  "outdir": "./models"
}

# ray.init(address="auto")
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
    num_samples=2,
    reuse_actors=True,
  ),
  run_config=RunConfig(
    stop=stopper,
  ),
)

results = tuner.fit()
best_result = results.get_best_result("val_accuracy", mode="max")
best_config = best_result.config
print("Best config:", best_config)

ray.shutdown()