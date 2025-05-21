# Sign Language Alphabet Classifier

This project builds a lightweight Convolutional Neural Network (CNN) to classify American Sign Language (ASL) alphabet gestures from grayscale 28x28 images. It uses a custom MNIST-style dataset and is optimized for fast training and Android deployment.

---

## 🚀 Features

- **PyTorch-based CNN**: Lightweight and fast for mobile inference.
- **Validation Split & Early Stopping**: Built-in to avoid overfitting.
- **TensorBoard & W&B Integration**: Track training visually.
- **Dataset Version Logging**: Track dataset version
- **TorchScript & ONNX Export**: Ready for mobile and cross-platform deployment.
- **Live Webcam Inference**: Predict ASL signs from webcam feed.

---

## 📁 Project Structure

- `train.py`: Main training loop (loss, validation, accuracy, early stopping, saving).
- `trainer.py`: Functions for training, validation, and evaluation.
- `model.py`: CNN model definition.
- `utils.py`: Dataset class and data loader utilities.
- `export.py`: Exports trained model to TorchScript and ONNX formats.
- `webcam_demo.py`: Real-time webcam-based inference demo.
- `trained_models/`: Saved PyTorch models.
- `runs/`: TensorBoard logs.
- `README.md`: Project documentation.

---

## 🧠 Dataset

The dataset is a CSV version of sign language images modeled after MNIST obtained from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data):

- Images are 28x28 grayscale pixels, flattened into 784 columns.
- Labels are integers (0–25), corresponding to ASL alphabets A–Z, **excluding J (9)** and **Z (25)** due to motion requirements.
- CSV format: `label, pixel1, pixel2, ..., pixel784`.

Download:
- `sign_mnist_train.csv`
- `sign_mnist_test.csv`

---

## ⚙️ Setup Instructions

1. **Clone the repo**:
   - `git clone https://github.com/yourusername/MNIST_sign.git && cd MNIST_sign`

2. **Create a Python environment**:
   - `conda create -n signlang python=3.10 -y && conda activate signlang`

3. **Install dependencies**:
   - `pip install -r requirements.txt`

---

## 🏋️ Training

Run the training script. Example:
```bash
python train.py --use_wandb --batch_size 128 --epochs 20
```
- Performs training with validation split.
- Saves best model (`trained_models/sign_cnn_best.pth`).
- Logs losses/accuracy to TensorBoard.

Launch TensorBoard:
```bash
tensorboard --logdir=runs
```

Open the provided URL in your browser to view training stats.

---

## 📦 Export Models

To export the best trained model to TorchScript and ONNX:
```bash
python export.py
```
- TorchScript saved as: `sign_model_scripted.pt`
- ONNX saved as: `sign_model.onnx`

These models are optimized for Android or cross-platform deployment.

---

## 🎥 Webcam Demo

To try real-time prediction using your webcam:
```bash
python webcam_demo.py
```
- Model must be trained or loaded beforehand.
- The window shows predicted ASL class for the current frame.
- Press `q` to exit.

---

## 📱 Android Integration (Next Step)

You can deploy the `sign_model_scripted.pt` or `sign_model.onnx` model to Android using:
- PyTorch Mobile
- ONNX Runtime for Android

Integrate the exported model into an Android app using the appropriate runtime library.

---

## 🌐 API Inference (Flask & FastAPI)

You can serve the trained model using either **Flask** or **FastAPI** for web/API-based inference.

### 🧪 Flask Inference API

Run the Flask app locally:

```bash
python flask_app.py
```

- Accepts image uploads via POST.
- Returns predicted ASL character.
- Endpoint: http://localhost:5000/predict

Example request (using curl):

```bash 
curl -X POST -F image=@sample.png http://localhost:5000/predict
```

### ⚡ Flask Inference API

```bash 
uvicorn app.main:app --reload
```

## 🐳 Docker Support

Run the entire app (model + API) in a Docker container.

### 🏗️ Build the image

```bash 
docker build -t signlang-api .
```

### 🚀 Run the container

```bash 
docker run -p 8000:8000 signlang-api
```