import torch
from torchvision import transforms
from PIL import Image


def load_model(model_path="trained_models/sign_model_scripted.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, device


if __name__ == "__main__":
    from utils import predict_image

    model, device = load_model()
    img = Image.open("test_image.jpg")
    label, conf = predict_image(img, model, device)
    print(f"Predicted: {label}, Confidence: {conf:.2f}%")
