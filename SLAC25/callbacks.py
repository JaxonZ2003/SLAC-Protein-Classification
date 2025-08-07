class BaseCallback:
  def on_fit_start(self, learner):
    """Call at the beginning of the training process"""
    pass

  def on_epoch_end(self, learner, epoch, logs):
    """Call at the end of the epoch"""
    pass

  def on_fit_end(self, learner):
    """Call at the end of the training process"""
    pass

class LoggingCallback(BaseCallback):
  def on_fit_start(self, learner):
    print("--- Start Training ---")
  
  def on_epoch_end(self, learner, epoch, logs):
    # The 'logs' dictionary is passed from the learner
    train_loss = logs.get('train_loss', 'N/A')
    val_loss = logs.get('val_loss', 'N/A')
    metrics = logs.get('metrics', {})

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
    for name, value in metrics.items():
      if value.numel() == 1: # For scalar tensors
          print(f"  -> {name}: {value.item():.4f}")
      else: # For multi-value tensors
          print(f"  -> {name}: {value.tolist()}")
  
  def on_fit_end(self, learner):
    print("--- Finish Training ---")