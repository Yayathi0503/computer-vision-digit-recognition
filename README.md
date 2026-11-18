# Handwritten digit recognition

Classify a single centered digit from a PNG or JPEG with a support vector machine. Pillow converts the image to grayscale and fits it inside an 8×8 square before classification.

## Run the demo

Use Python 3.12. From the repository root, create a virtual environment with `python -m venv .venv`. Activate it (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux), then run:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

```sh
python digits.py
python digits.py my-digit.png
python digits.py dark-ink-on-white.png --invert
```

The first command trains on the bundled scikit-learn digits dataset and prints holdout accuracy plus a 10×10 confusion matrix (rows are true digits, columns are predictions). Each invocation trains in memory. An image command prints one digit from 0 through 9.

## Input and evaluation

Crop closely around one centered digit. The default input is a light digit on a dark background. The model trains on 1,347 small grayscale samples with a stratified 450-image holdout, seed 42. Photographs, full pages, and handwriting from a different source can give substantially different results.
