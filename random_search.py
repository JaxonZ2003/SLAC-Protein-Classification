from ray import tune

from ray.tune import Tuner, TuneConfig, RunConfig, CheckpointConfig, FailureConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.execution.placement_groups import PlacementGroupFactory

from SLAC25.tune import Trainable
from search_space import Experiments

trainable_with_resources = tune.with_resources(
    Trainable,
    resources=PlacementGroupFactory([{"CPU": 1, "GPU": 1}])
)

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='val_accuracy',
    mode='max',
    max_t=30,                       # largest running epochs
    grace_period=5,
    reduction_factor=3,             # keeps 1/3 of the trials eventually
    brackets=1,
)

tuner = Tuner(
  trainable_with_resources,
  tune_config=TuneConfig(
  metric="val_accuracy",
  mode="max",
  num_samples=10, # if grid_search, keep it =1 if wanna run all combination
  scheduler=asha_scheduler,
  reuse_actors=False,
),run_config=RunConfig(
  name=exp_name,
  stop=combinedStopper,
  storage_path=storage_path,
  progress_reporter=reporter,
  checkpoint_config=CheckpointConfig(
  # write ONLY at the very end
  checkpoint_frequency=0,
  checkpoint_at_end=True,
  num_to_keep=1,                       # keep one file max
  failure_config=FailureConfig(max_failures=3, 
                              fail_fast=False)
  )
)