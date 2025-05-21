import torch
from torchvision import transforms
from PIL import Image

classes = [
    chr(i) for i in range(65, 91) if i != ord("J") and i != ord("Z")
]  # A-Z without J, Z

transform = transforms.Compose(
    [
        transforms.Grayscale(),  # Ensure single channel if needed
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def predict_image(image: Image.Image, model, device: torch.device):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    return int(pred.item()), float(confidence.item()) * 100
