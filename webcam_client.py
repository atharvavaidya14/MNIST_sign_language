import cv2
import httpx
import asyncio
import time
from typing import Dict, Any

url = "http://localhost:8000/predict"


async def send_frame(frame) -> Dict[str, Any]:
    """Send a frame to the server for prediction."""
    _, img_encoded = cv2.imencode(".jpg", frame)
    try:
        start = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                timeout=5.0,
            )
        duration = (time.time() - start) * 1000  # ms
        print(f"Inference time: {duration:.2f} ms")
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"HTTP error: {e}")
        return {"label": "Error", "confidence": 0}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"label": "Error", "confidence": 0}


async def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))

        result = await send_frame(resized)

        label = result.get("label", "N/A")
        confidence = result.get("confidence", None)
        text = f"{label} ({confidence:.2f}%)" if confidence else label

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Language Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
