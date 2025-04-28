# script to test the best hyperparams with a simple training loop for simplicity

from SLAC25.models import ResNet
from SLAC25.network import ModelWrapper
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

class HyperparameterTuner:
    def __init__(self, search_space="small"):
        # Define the search spaces
        self.search_spaces = {
            "large": {
                "hidden_dim": tune.choice([256, 512]),
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "keep_prob": tune.uniform(0.5, 0.9),
                "beta1": tune.uniform(0.5, 0.9),
                "beta2": tune.uniform(0.5, 0.9),
            },
            "small": {
                "hidden_dim": tune.choice([64, 128]),
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "keep_prob": tune.uniform(0.5, 0.9),
                "beta1": tune.uniform(0.5, 0.9),
                "beta2": tune.uniform(0.5, 0.9),
            }
        }
        
        self.search_space = self.search_spaces[search_space]
        
        # Configure Hyperband scheduler
        self.scheduler = HyperBandScheduler(
            time_attr="training_iteration",
            max_t=10,
            reduction_factor=3,
            metric="loss",
            mode="min"
        )
    
    def objective(self, config, checkpoint_dir=None):
        '''
        training loop for finding the best hyperparameters using Ray Tune Hyperband scheduler
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training model with config: ", config)

        # load data and model
        model = ResNet(num_classes=4, keep_prob=config["keep_prob"], hidden_dim=config["hidden_dim"])
        wrapper = ModelWrapper(model=model, verbose=True, testmode=False)        
        train_loader, test_loader, val_loader = wrapper._prepareDataLoader(batch_size=16, testmode=False, max_imgs=5000, nwork=4)
        criterion = nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config["learning_rate"], 
            betas=(config["beta1"], config["beta2"])
        )
        
        # Train the model
        print("Training model...")
        for epoch in range(10):
            # training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = val_correct / val_total
            val_loss = val_loss / len(val_loader)

            tune.report({"loss": val_loss, "mean_accuracy": val_acc})
            print("Epoch {}/10, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch+1, val_loss, val_acc))
    
    def run_tuning(self, num_samples=10, max_epochs=10):
        """
        Run the hyperparameter tuning process
        args:
            num_samples: number of hyperparameter configurations to try
        """
        tuner = tune.Tuner(
            self.objective,
            # configure the tuning process
            tune_config=tune.TuneConfig(
                scheduler=self.scheduler,
                num_samples=num_samples,
            ),
            param_space=self.search_space,
            # configure the run
            run_config=tune.RunConfig(
                name="hyperband_test",
                stop={"training_iteration": max_epochs},
                verbose=1,
            )
        )
        results = tuner.fit()
        
        # Get and return the best result
        best_result = results.get_best_result(metric="loss", mode="min")
        return best_result

# Example usage:
if __name__ == "__main__":
    # add in argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_space", type=str, default="small", choices=["small", "large"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # run the tuning process
    tuner = HyperparameterTuner(search_space=args.search_space)
    best_result = tuner.run_tuning(num_samples=args.samples, max_epochs=args.epochs)
    print("Best hyperparameters found were: ", best_result.config)