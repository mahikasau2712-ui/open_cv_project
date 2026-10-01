import argparse

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate facial expressions with OpenCV.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    return parser.parse_args()


def classify_expression(face_gray, face_width: int, face_height: int) -> str:
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eyes = eye_cascade.detectMultiScale(
        face_gray[: face_height // 2],
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(max(face_width // 10, 10), max(face_height // 10, 10)),
    )
    smiles = smile_cascade.detectMultiScale(
        face_gray[face_height // 2 :],
        scaleFactor=1.7,
        minNeighbors=22,
        minSize=(max(face_width // 5, 20), max(face_height // 10, 15)),
    )
    if len(smiles) > 0:
        return "Happy"
    if len(eyes) >= 2 and face_height > face_width * 1.05:
        return "Surprised"
    return "Neutral"


def main() -> None:
    args = parse_args()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("OpenCV face cascade could not be loaded.")
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Try --camera 1 or check permissions."
        )

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            for x, y, width, height in faces:
                face_gray = gray_frame[y : y + height, x : x + width]
                expression = classify_expression(face_gray, width, height)
                color = (0, 200, 0) if expression == "Happy" else (0, 180, 255)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(
                    frame,
                    expression,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            cv2.imshow("Face Expression Reading", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
