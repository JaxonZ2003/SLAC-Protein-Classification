from ray import tune

'''
Using Stratified Sampler of 
'''

# Experiment 1
'''
Tune for the best Learning Rate of BaselineCNN
'''
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

# Experiment 2
'''
Tune for the best hyperparams of ResNet that cause significant improvement of Loss
'''
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