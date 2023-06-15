import numpy as np


def test_custom_transform():
    from src import load_data
    from src import custom_transform
    from src import Perceptron

    X, y, _ = load_data("data/spiral.csv")

    new_X = custom_transform(X)
    msg = "Only use at most three features"
    assert new_X.shape[1] <= 3, msg

    model = Perceptron()
    model.fit(new_X, y)
    preds = model.predict(new_X)
    acc = np.mean(preds == y)
    msg = f"Need {100 * acc:.1f}% accuracy"
    assert acc >= 0.9, msg
