from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.app.app_utils import load_serialized_model
from src.app.app_utils import predict_image
from PIL import Image
import io

app = FastAPI()
model, device = load_serialized_model()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict the label of an image."""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    label, confidence = predict_image(image, model, device)
    return JSONResponse(content={"label": label, "confidence": confidence})
