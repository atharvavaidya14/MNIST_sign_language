import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from src.models.model_architecture import SimpleCNN
import torch.nn.functional as F
from typing import Tuple

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(
    torch.load("trained_models\sign_cnn_best.pth", map_location=device)
)
model.eval()

# Define transform (same as training!)
transform = transforms.Compose(
    [
        transforms.Grayscale(),  # Ensure it's single-channel
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


# Inference function
def predict(image_path: str) -> Tuple[int, float]:
    """
    Predict the label of an image and return (predicted_class, confidence %).
    """
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return int(predicted.item()), float(confidence.item()) * 100


# Example usage
if __name__ == "__main__":
    img_path = r"test_image.jpg"  # replace with your image
    class_idx, conf = predict(img_path)
    print("Predicted label index:", class_idx)
    print("Confidence (%):", round(conf * 100, 2))
    print("Predicted character:", chr(class_idx + ord("A")))
