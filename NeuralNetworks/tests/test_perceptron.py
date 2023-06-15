import numpy as np 


def test_perceptron():
    from src import Perceptron, load_data
    from sklearn.metrics import accuracy_score

    features, targets, _ = load_data('data/parallel-lines.csv')
    max_iter = 100
    p = Perceptron(max_iter=max_iter)
    num_iter = p.fit(features, targets)
    targets_hat = p.predict(features)

    # your perceptron should fit this dataset perfectly
    assert accuracy_score(targets, targets_hat) == 1.0
    assert num_iter < max_iter


def test_polynomial_perceptron():
    from src import Perceptron, load_data
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import PolynomialFeatures

    features, targets, _ = load_data('data/circles.csv')
    max_iter = 1000
    p = Perceptron(max_iter=max_iter)
    num_iter = p.fit(features, targets)
    targets_hat = p.predict(features)
    assert accuracy_score(targets, targets_hat) < 1.0

    # after polynomial transform, should fit perfectly
    poly_features = PolynomialFeatures(2).fit_transform(features)
    num_iter = p.fit(poly_features, targets)
    targets_hat = p.predict(poly_features)
    assert accuracy_score(targets, targets_hat) == 1.0
