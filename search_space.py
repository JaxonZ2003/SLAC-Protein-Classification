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
  "seed": 197,
  "tune": True,
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
  "seed": 197,
  "tune": True,
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
  "seed": 197,
  "tune": True,
}


# Experiment 4
'''
Tune for the best learning rate of ResNet50 with hidden dimension of 256
'''
ResNet_lr256_sp = {
  "name": "ResNet_dim256_lr",
  "model": "ResNet",
  "batch_size": 32,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 256,
  "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
  "keep_prob": 0.5,
}

# Experiment 5
'''
Tune for the best learning rate of ResNet50 with hidden dimension of 512
'''
ResNet_lr512_sp = {
  "name": "ResNet_dim512_lr",
  "model": "ResNet",
  "batch_size": 32,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 512,
  "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
  "keep_prob": 0.5
}

# Experiment 6
'''
Tune for the best drop out rate of ResNet50 with hidden dimension of 256
'''
ResNet_dr256_lr4 = {
  "name": "ResNet_dr256_lr4",
  "model": "ResNet",
  "batch_size": 32,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 256,
  "lr": 1e-4,
  "keep_prob": tune.grid_search([0.35, 0.45, 0.55, 0.65, 0.75])
}

# Experiment 7
'''
Tune for the effect of learning rate scheduler of ResNet50 with hidden dimension of 256
'''
ResNet_sl256_lr4 = {
  "name": "ResNet_sl256_lr4",
  "model": "ResNet",
  "batch_size": 32,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 256,
  "lr": 1e-4,
  "keep_prob": 0.5,
  "lr_scheduler": tune.grid_search([True, False])
}

# Experiment 8
'''
Tune for the beta1 of ResNet50 with hidden dimension of 256
'''
ResNet_beta1 = {
  "name": "ResNet_beta1",
  "model": "ResNet",
  "batch_size": 32,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 256,
  "lr": 1e-4,
  "keep_prob": 0.5,
  "lr_scheduler": True,
  "beta1": tune.grid_search([0.5, 0.7, 0.9, 0.95])
}

# Experiment 9
'''
Tune for the beta2 of ResNet50 with hidden dimension of 256
'''
ResNet_beta2 = {
  "name": "ResNet_beta2",
  "model": "ResNet",
  "batch_size": 32,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 256,
  "lr": 1e-4,
  "keep_prob": 0.5,
  "lr_scheduler": True,
  "beta1": 0.9,
  "beta2": tune.grid_search([0.9, 0.99, 0.999, 0.9999])
}


# Experiment 10
'''
Train the best parameters so far after grid search from experiment 4 to 10
'''
BestSoFarR = {
  "name": "BestSoFarR",
  "model": "ResNet",
  "batch_size": 16,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 256,
  "lr": 1e-4,
  "keep_prob": 0.55,
  "lr_scheduler": True,
  "beta1": 0.9,
  "beta2": 0.999,
  "tune": False,
}


############ Random Search ASHAS ############

# Experiment 11
'''
Finer stochastic search of parameters in ResNet50
'''

RandomSearch_1 = {
  "name": "RandomSearch_1",
  "model": "ResNet",
  "scheduler": "ASHA",
  "batch_size": 32,
  "testmode": False,
  "seed": 197,
  "hidden_dim": 256,
  "lr": tune.loguniform(1e-5, 1e-4),
  "keep_prob": tune.quniform(0.75, 1, 0.01),
  "lr_scheduler": True,
  "beta1": tune.uniform(0.5, 0.99),
  "beta2": tune.quniform(0.99, 0.99999, 0.00001),
  "tune": True,
}


Experiments = {
  "1": BaselineCNN_lr_sp,
  "2": BaselineCNN_dr_sp,
  "3": BaselineCNN_sl_sp,
  "4": ResNet_lr256_sp,
  "5": ResNet_lr512_sp,
  "6": ResNet_dr256_lr4,
  "7": ResNet_sl256_lr4,
  "8": ResNet_beta1,
  "9": ResNet_beta2,
  "10": BestSoFarR,
  "11": RandomSearch_1
}