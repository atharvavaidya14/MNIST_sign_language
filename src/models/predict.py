import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from typing import Tuple
from src.utils.utils import load_model, get_device

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
    model = load_model()
    device = get_device()
    image = Image.open(image_path).convert("L")  # Convert to grayscale
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
