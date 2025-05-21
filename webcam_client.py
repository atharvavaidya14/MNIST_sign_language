import cv2
import requests

API_URL = "http://localhost:5000/predict"

cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to JPG bytes
    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {"image": img_encoded.tobytes()}

    # Send to API
    response = requests.post(API_URL, files=files)
    if response.ok:
        data = response.json()
        pred = data["prediction"]
        conf = data["confidence"]
        cv2.putText(
            frame,
            f"{pred} ({conf:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Sign Language Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
