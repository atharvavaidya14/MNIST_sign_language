from flask import Flask, request, jsonify
import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.models.model_architecture import SimpleCNN

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model weights
model = SimpleCNN().to(DEVICE)
model.load_state_dict(
    torch.load("trained_models/sign_cnn_best.pth", map_location=DEVICE)
)
model.eval()

# Define the same transforms used during training
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Class labels
CLASS_NAMES = [chr(i) for i in range(65, 91)]  # A-Z


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the label of an image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    predicted_letter = CLASS_NAMES[pred_idx.item()]
    confidence = conf.item()

    return jsonify({"prediction": predicted_letter, "confidence": confidence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
