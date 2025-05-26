import io
import pytest
from PIL import Image
from src.app.flask_app import app  # Import the actual Flask app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def create_dummy_image():
    img = Image.new("RGB", (28, 28), color=(255, 255, 255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_predict_success(client):
    img_bytes = create_dummy_image()
    data = {"image": (img_bytes, "dummy.jpg")}
    response = client.post("/predict", content_type="multipart/form-data", data=data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert "prediction" in json_data
    assert "confidence" in json_data


def test_missing_image(client):
    response = client.post("/predict", content_type="multipart/form-data", data={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "No image file provided"


def test_invalid_file_format(client):
    bad_bytes = io.BytesIO(b"not an image")
    data = {"image": (bad_bytes, "fake.txt")}
    response = client.post("/predict", content_type="multipart/form-data", data=data)
    assert response.status_code in [400, 500]  # Depending on your error handling
