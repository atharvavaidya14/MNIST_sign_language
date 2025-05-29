# Sign Language Alphabet Classifier (End-to-end)

This project is an **end-to-end machine learning pipeline** for classifying American Sign Language (ASL) alphabets from grayscale 28x28 images. It includes building a lightweight CNN model, training it, exporting models, deploying an API, and automating CI/CD using Docker, GitHub Actions, and Render.

---

## 🔧 Features

- **PyTorch-based CNN**: Lightweight and fast for mobile inference.
- **Validation Split & Early Stopping**: Built-in to avoid overfitting.
- **TensorBoard & W&B Integration**: Track training visually.
- **Dataset Version Logging**: Track dataset version
- **TorchScript & ONNX Export**: Ready for mobile and cross-platform deployment.
- **Live Webcam Inference**: Predict ASL signs from webcam feed.
- **Flask & FastAPI APIs** for web-based inference.
- **Dockerized Deployment** and **CI/CD with GitHub Actions**.
- **Render-based Cloud Deployment** using Docker image.
- **Integrated Testing and Formatting Pipelines** with Black, Flake8, and Pytest.

---

## 📁 Project Structure

- `src/training/train.py`: Main training loop (loss, validation, accuracy, early stopping, saving).
- `src/training/trainer.py`: Functions for training, validation, and evaluation.
- `src/models/model_architecture.py`: CNN model definition.
- `src/utils/utils.py`: Dataset class and data loader utilities.
- `src/models/export.py`: Exports trained model to TorchScript and ONNX formats.
- `src/demo/webcam_demo.py`: Real-time webcam-based inference demo.
- `trained_models/`: Saved PyTorch models.

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

```bash
git clone https://github.com/atharvavaidya14/MNIST_sign_language.git && cd MNIST_sign_language
conda create -n signlang python=3.10 -y && conda activate signlang
pip install -r requirements.txt
```

---

## 🏋️ Model Training

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

## 🌐 API Inference (Flask & FastAPI)

You can serve the trained model using either **Flask** or **FastAPI** for web/API-based inference.

### 🧪 Flask Inference API

Run the Flask app locally:

```bash
python flask_app.py
```

- Accepts image uploads via POST requests.
- Returns predicted ASL character.
- Endpoint: <http://localhost:5000/predict>

Example request (using curl):

```bash
curl -X POST -F image=@sample.png http://localhost:5000/predict
```

### ▶️ FastAPI (Recommended)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- `POST /predict`: Accepts image file, returns predicted letter.

---

## 🐳 Docker Deployment

### 🏗️ Build

```bash
docker build -t atharvavaidya14/sign-lang-app:latest .
```

### 🏃 Run

```bash
docker run -p 8000:8000 atharvavaidya14/sign-lang-app:latest
```

---

## 🔁 CI/CD Pipeline

### 🚲 CI (GitHub Actions)

- Checks for code formatting (Black), linting (Flake8), and unit tests (Pytest).
- Builds Docker image on push or PR to `deploy` branch.
- Pushes image to DockerHub.

### ✈️ CD (Render)

- Uses DockerHub image.
- Automatically deploys updated image.

### 🔗 URL

<https://sign-lang-app.onrender.com>

⚠️ Ensure `POST /predict` is used instead of GET.

---

## ✅ Future Improvements

- Android app integration.
- Add frontend web interface.
- Multi-language sign support.
- Advanced sign language translator (more than alphabets and with video support)
