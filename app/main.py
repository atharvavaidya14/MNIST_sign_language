from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.model import load_model
from app.utils import predict_image
from PIL import Image
import io

app = FastAPI()
model, device = load_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    label, confidence = predict_image(image, model, device)
    return JSONResponse(content={"label": label, "confidence": confidence})
