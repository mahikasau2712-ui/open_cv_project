import argparse

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people with OpenCV HOG.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--confidence", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
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
            frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
            boxes, weights = detector.detectMultiScale(
                frame, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
            for (x, y, width, height), weight in zip(boxes, weights):
                confidence = float(weight)
                if confidence < args.confidence:
                    continue
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"Person {confidence:.2f}",
                    (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 0),
                    2,
                )
            cv2.imshow("OpenCV Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
