import numpy as np
from src import mean_squared_error as src_mse

def make_fake_data():
    y_true = np.random.random(100)
    y_pred = np.random.random(100)
    return y_pred, y_true

def test_mean_squared_error():
    from sklearn.metrics import mean_squared_error as sklearn_mse 
    y_pred, y_true = make_fake_data()

    _est = src_mse(y_true, y_pred)
    _actual = sklearn_mse(y_true, y_pred)

    assert np.allclose(_actual, _est)
