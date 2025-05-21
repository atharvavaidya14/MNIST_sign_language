import cv2
import requests

url = "http://localhost:8000/predict"

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_encoded = cv2.imencode(".jpg", gray)

    response = requests.post(url, files={"file": img_encoded.tobytes()})

    if response.ok:
        result = response.json()
        label = result["label"]
        conf = result["confidence"]
        cv2.putText(
            frame,
            f"{label} ({conf})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Live Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
