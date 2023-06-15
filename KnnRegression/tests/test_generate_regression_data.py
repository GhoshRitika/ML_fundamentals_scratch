import numpy as np

from src import generate_regression_data

def test_generate_regression_data():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    degrees = range(2, 5)
    n_examples = [10, 100, 1000, 10000]
    noise_amounts = [0, 1, 2]

    for degree in degrees:
        good_transform = PolynomialFeatures(degree)
        bad_transform = PolynomialFeatures(degree - 1)
        model = LinearRegression()

        for n in n_examples:
            prev_good_mse = -1
            for amount_of_noise in noise_amounts:
                np.random.seed(0)
                x, y = generate_regression_data(degree, n, amount_of_noise=amount_of_noise)
                assert (len(x) == n and len(y) == n), "Incorrect amount of data"
                assert (x.min() >= -1 and x.max() <= 1), "X data outside of [-1, 1] range"

                model.fit(good_transform.fit_transform(x), y)
                good_mse = mean_squared_error(y, model.predict(good_transform.fit_transform(x)))
                msg = f"With degree {degree} and {n} examples, expected {good_mse:.3f} > {prev_good_mse:.3f}"
                assert good_mse > prev_good_mse, msg
                prev_good_mse = good_mse

                model.fit(bad_transform.fit_transform(x), y)
                bad_mse = mean_squared_error(y, model.predict(bad_transform.fit_transform(x)))
                msg = f"With {degree} and {n} examples, expected {bad_mse:.3f} > {good_mse:.3f}"
                assert bad_mse > good_mse, msg

                if amount_of_noise == 0:
                    assert np.isclose(good_mse, 0), "With no noise, {good_mse:.3f} should be 0"
