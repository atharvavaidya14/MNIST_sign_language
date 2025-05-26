from PIL import Image
from src.app.app_utils import load_serialized_model, predict_image


if __name__ == "__main__":

    model, device = load_serialized_model()
    img = Image.open("test_image.jpg")
    label, conf = predict_image(img, model, device)
    print(f"Predicted: {label}, Confidence: {conf:.2f}%")
