import src
import numpy as np


def test_euclidean_distances():
    from sklearn.metrics.pairwise import euclidean_distances
    x = np.random.rand(100, 100)
    y = np.random.rand(100, 100)
    _true = euclidean_distances(x, y)
    _est = src.euclidean_distances(x, y)
    assert (np.allclose(_true, _est))


def test_manhattan_distances():
    from sklearn.metrics.pairwise import manhattan_distances
    x = np.random.rand(100, 100)
    y = np.random.rand(100, 100)
    _true = manhattan_distances(x, y)
    _est = src.manhattan_distances(x, y)
    assert (np.allclose(_true, _est))


def test_cosine_distances():
    from sklearn.metrics.pairwise import cosine_distances
    x = np.random.uniform(-1, 1, size=[100, 100])
    y = np.random.uniform(-1, 1, size=[100, 100])
    _true = cosine_distances(x, y)
    _est = src.cosine_distances(x, y)
    assert (np.allclose(_true, _est))

    for multiplier in [10, 100, 1000]:
        y_ = y * multiplier
        _est = src.cosine_distances(x, y_)
        assert (np.allclose(_true, _est))
