import numpy as np
from PIL import Image
import pytest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from digits import classify, experiment


def test_holdout_report():
    _, report = experiment()
    matrix = np.array(report["confusion_matrix"])
    assert matrix.shape == (10, 10)
    assert matrix.sum() == report["test_images"] == 450
    assert matrix.trace() / matrix.sum() == report["accuracy"] > 0.90


@pytest.mark.parametrize("invert", [False, True])
def test_classifies_encoded_holdout_images(tmp_path, invert):
    data = load_digits()
    _, indices = train_test_split(np.arange(len(data.target)), test_size=0.25,
                                  stratify=data.target, random_state=42)
    correct = 0
    for index in indices[:20]:
        pixels = (data.images[index] * 255 / 16).round().astype("uint8")
        path = tmp_path / "sample.png"
        Image.fromarray(255 - pixels if invert else pixels).save(path)
        correct += classify(path, invert) == data.target[index]
    assert correct >= 18


def test_blank_rejected(tmp_path):
    path = tmp_path / "empty.png"
    Image.new("L", (64, 64)).save(path)
    with pytest.raises(ValueError, match="uniform"):
        classify(path)
