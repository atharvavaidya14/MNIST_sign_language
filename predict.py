import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import SimpleCNN

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("sign_cnn.pth", map_location=device))
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
def predict(image_path):
    image = Image.open(image_path).convert("L")  # Grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dim
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.argmax(outputs, dim=1).item()
    return predicted


# Example usage
if __name__ == "__main__":
    img_path = r"WIN_20250520_17_48_29_Pro.jpg"  # replace with your image
    class_idx = predict(img_path)
    print("Predicted label index:", class_idx)
    print("Predicted character:", chr(class_idx + ord("A")))
