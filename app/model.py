import torch
from model import SimpleCNN  # Use your original model
import os


def load_model():
    model = SimpleCNN()
    checkpoint = "trained_models/sign_cnn_best.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device
