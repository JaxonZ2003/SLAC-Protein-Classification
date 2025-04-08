import ray
import json

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.tuner import Tuner

from SLAC25.utils import *
from SLAC25.network import ModelWrapper
from SLAC25.models import * # import the model

with open('config.json', 'r') as f:
  raw_config = json.load(f)

# print(raw_config)

search_space = {
  "model": tune.grid_search(raw_config["model"]),
  "num_classes": tune.grid_search(raw_config["num_classes"]),
  "keep_prob": tune.grid_search(raw_config["keep_prob"]),
  "num_epochs": tune.grid_search(raw_config["num_epochs"]),
  "batch_size": tune.grid_search(raw_config["batch_size"]),
  "verbose": tune.grid_search(raw_config["verbose"]),
  "testmode": tune.grid_search(raw_config["testmode"]),
}

# ray.init(address="auto")
ray.init()

model = ResNet(num_classes=4, keep_prob=0.75)

model_wrapper = ModelWrapper(model, verbose=True, testmode=True)
tuner = Tuner(
  model_wrapper.train,
  param_space=search_space,
  tune_config=tune.TuneConfig(max_concurrent_trials=2),
  # run_config=air.RunConfig(resources_per_trial={"cpu": 2, "gpu": 1})
)

results = tuner.fit()
best_result = results.get_best_result("mean_accuracy", mode="max")

with best_result.checkpoint.as_directory() as checkpoint_dir:
  pass

print(ray.__version__)