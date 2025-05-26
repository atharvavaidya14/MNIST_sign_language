from flask import Flask, request, jsonify
import io
import logging
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.models.model_architecture import SimpleCNN
from src.utils.utils import get_device

# ----------------------------- Logging Setup -----------------------------
logging.basicConfig(
    filename="flask_app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ----------------------------- App and Model Setup -----------------------------
app = Flask(__name__)
DEVICE = get_device()

try:
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(
        torch.load("trained_models/sign_cnn_best.pth", map_location=DEVICE)
    )
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.exception("Failed to load model:")
    raise RuntimeError("Model loading failed. See logs for details.") from e

# ----------------------------- Transform & Class Labels -----------------------------
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

CLASS_NAMES = [chr(i) for i in range(65, 91)]  # A-Z


# ----------------------------- Routes -----------------------------
@app.route("/")
def index():
    return "Welcome to the Sign Language API"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        logging.warning("Request received without an image file.")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        logging.exception("Image preprocessing failed:")
        return jsonify({"error": f"Invalid image format. Details: {str(e)}"}), 400

    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        predicted_letter = CLASS_NAMES[pred_idx.item()]
        confidence = conf.item()

        logging.info(f"Prediction successful: {predicted_letter} ({confidence:.2f})")
        return jsonify({"prediction": predicted_letter, "confidence": confidence})
    except Exception as e:
        logging.exception("Model inference failed:")
        return jsonify({"error": f"Inference failed. Details: {str(e)}"}), 500


# ----------------------------- Run -----------------------------
if __name__ == "__main__":
    logging.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000)
