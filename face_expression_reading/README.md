# Face Expression Reading

Webcam face detection with a simple visible-feature estimate: `Happy`, `Surprised`, or `Neutral`.

```cmd
python face_expression_reading\face_expression_reading.py
python face_expression_reading\face_expression_reading.py --camera 1
```

It uses Haar cascades shipped with OpenCV, so no separate model download is needed. Press `q` to close the window. This is an educational estimate and should not be treated as a reliable measurement of a person's feelings.
