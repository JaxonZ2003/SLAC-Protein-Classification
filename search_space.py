from ray import tune

'''
Using Stratified Sampler
'''

# Experiment 1
'''
Tune for the best Learning Rate of BaselineCNN
'''
BaselineCNN_lr_sp = {
  "name": "BaselineCNN_lr", 
  "model": "BaselineCNN",
  "keep_prob": tune.grid_search([0.75]),
  "lr": tune.grid_search([0.001, 0.01, 0.1]),
  "batch_size": 32,
  "testmode": False,
  "lr_scheduler": False,
  "seed": 197
}

# Experiment 2
'''
Tune for the best dropout rate of BaselineCNN
'''
BaselineCNN_dr_sp = {
  "name": "BaselineCNN_dropout", 
  "model": "BaselineCNN",
  "keep_prob": tune.grid_search([0.35, 0.45, 0.55, 0.65, 0.75]),
  "lr": 0.001,
  "batch_size": 32,
  "testmode": False,
  "lr_scheduler": False,
  "seed": 197
}

# Experiment 3
'''
Tune for the effect of lr_scheduler
'''
BaselineCNN_sl_sp = {
  "name": "BaselineCNN_scheduler", 
  "model": "BaselineCNN",
  "keep_prob": 0.55,
  "lr": 0.001,
  "batch_size": 32,
  "testmode": False,
  "lr_scheduler": tune.grid_search([True, False]),
  "seed": 197
}


# Experiment 4
'''
Tune for the best hyperparams of ResNet that cause significant improvement of Loss
'''
ResNet_lr256_sp = {
  "model": "ResNet",
  "batch_size": 32,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 256,
  "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
  "keep_prob": 0.5
}

# Experiment 5
'''
Tune for the best hyperparams of ResNet that cause significant improvement of Loss
'''
ResNet_lr512_sp = {
  "model": "ResNet",
  "batch_size": 32,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 512,
  "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
  "keep_prob": 0.5
}

# resnet_sp = {
#   "model": "ResNet",
#   "batch_size": 32,
#   "testmode": False,
#   "outdir": "./models",
#   "tune": True,
#   "seed": 197,
#   "hidden_dim": tune.choice([256, 512]),
#   "lr": tune.choice(1e-4, 1e-1),
#   "keep_prob": tune.uniform(0.5, 0.9),
#   "beta1": tune.uniform(0.5, 0.9),
#   "beta2": tune.uniform(0.5, 0.9),
# }

Experiments = {
  "1": BaselineCNN_lr_sp,
  "2": BaselineCNN_dr_sp,
  "3": BaselineCNN_sl_sp,
  "4": ResNet_lr256_sp,
  "5": ResNet_lr512_sp,
}