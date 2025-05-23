import torch
from src.models.model_architecture import SimpleCNN

def test_model_forward_pass():
    model = SimpleCNN()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 26)  # 26 classes (sign language letters)
