# OpenCV Vision Lab

Two computer-vision subprojects built with Python and OpenCV.

## Project structure

```text
OpenCV-Vision-Lab/
|-- object_detection/
|   |-- object_detection.py
|   `-- README.md
|-- face_expression_reading/
|   |-- face_expression_reading.py
|   `-- README.md
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Setup on Windows CMD

```cmd
cd /d "C:\path\to\OpenCV-Vision-Lab"
py -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` as the interpreter.

## Run

```cmd
python object_detection\object_detection.py
python face_expression_reading\face_expression_reading.py
```

Press `q` in the camera window to quit. Use `--camera 1` if needed.

## GitHub

```cmd
git init
git add .
git commit -m "Add OpenCV vision subprojects"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The object detector uses OpenCV's built-in HOG person detector. The expression project is an educational visible-feature estimate, not a medical or psychological assessment.
