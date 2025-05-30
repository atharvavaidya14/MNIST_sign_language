import cv2
import torch
from torchvision import transforms

# Label map (adjust for missing letters like 'J' and 'Z')
LABEL_MAP = [
    chr(i) for i in range(ord("A"), ord("Z") + 1) if i not in [ord("J"), ord("Z")]
]

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def load_serialized_model(
    path="trained_models/sign_model_scripted.pt",
) -> torch.jit.ScriptModule:
    """Load the pre-trained model."""
    model = torch.jit.load(path)
    model.eval()
    return model


def predict(model, frame) -> str:
    """Predict the sign language letter from the frame."""
    input_tensor = transform(frame).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return LABEL_MAP[predicted.item()]


def main():
    model = load_serialized_model()
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (28, 28))
        label = predict(model, roi)
        cv2.putText(
            frame,
            f"Prediction: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Sign Language Prediction", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
