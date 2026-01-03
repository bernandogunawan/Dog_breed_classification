"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch

def save_model(model, model_name):

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt")

  # Save the model state_dict()
  print(f"[INFO] Saving model")
  torch.save(obj=model.state_dict(), f = model_name)
