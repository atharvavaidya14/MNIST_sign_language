import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple


transform = transforms.Compose(
    [
        transforms.Grayscale(),  # Ensure single channel if needed
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def predict_image(image: Image.Image, model, device: torch.device) -> Tuple[int, float]:
    """
    Predict the label of an image using the trained model."""
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    return int(pred.item()), float(confidence.item()) * 100


def load_serialized_model(
    model_path="trained_models/sign_model_scripted.pt",
) -> Tuple[torch.jit.ScriptModule, torch.device]:
    """
    Load the pre-trained model from the specified path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, device
