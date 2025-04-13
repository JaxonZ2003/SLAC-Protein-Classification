import ray
import json

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.tuner import Tuner

from SLAC25.utils import *
from SLAC25.network import Trainable
from SLAC25.models import * # import the model

with open('config.json', 'r') as f:
  raw_config = json.load(f)

# print(raw_config)

search_space = {
  "model": ["BaselineCNN"],
  "num_classes": [4],
  "keep_prob": [0.75],
  "num_epochs": [3],
  "lr": [0.001],
  "batch_size": [32],
  "verbose": True,
  "testmode": True,
  "outdir": "./models"
}

# ray.init(address="auto")
ray.init()

# model = ResNet(num_classes=4, keep_prob=0.75)

# model_wrapper = ModelWrapper(model, verbose=True, testmode=True)
tuner = Tuner(
  Trainable,
  param_space=search_space,
  tune_config=tune.TuneConfig(max_concurrent_trials=1,
                              reuse_actors=True),
  run_config=ray.tune.RunConfig(
        verbose=0,
        checkpoint_config=ray.tune.CheckpointConfig(checkpoint_at_end=False),
    ),
)

results = tuner.fit()
best_result = results.get_best_result("mean_accuracy", mode="max")

with best_result.checkpoint.as_directory() as checkpoint_dir:
  pass

print(ray.__version__)