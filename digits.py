"""Evaluate an SVM or classify a single light-on-dark digit image."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


@lru_cache(maxsize=1)
def experiment():
    images, labels = load_digits(return_X_y=True)
    train, test, y_train, y_test = train_test_split(
        images / 16, labels, test_size=0.25, stratify=labels, random_state=42
    )
    model = SVC(C=4).fit(train, y_train)
    predictions = model.predict(test)
    return model, {"test_images": len(test), "accuracy": accuracy_score(y_test, predictions),
                   "confusion_matrix": confusion_matrix(y_test, predictions).tolist()}


def classify(path, invert=False):
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image)
        if invert:
            gray = ImageOps.invert(gray)
        if gray.getextrema()[0] == gray.getextrema()[1]:
            raise ValueError("A uniform image contains no visible digit")
        square = ImageOps.pad(gray, (8, 8), method=Image.Resampling.LANCZOS, color=0)
        features = np.asarray(square, dtype=float).reshape(1, 64) / 255
    return int(experiment()[0].predict(features)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("--invert", action="store_true", help="For dark ink on a light background")
    args = parser.parse_args()
    if args.image:
        try:
            print(classify(args.image, args.invert))
        except (OSError, ValueError) as exc:
            parser.exit(2, f"{exc}\n")
    else:
        print(json.dumps(experiment()[1], indent=2))
