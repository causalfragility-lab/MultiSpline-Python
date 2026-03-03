import numpy as np
import pandas as pd
from multispline import MultiSpline


def make_test_data(seed=42):
    np.random.seed(seed)
    n_schools = 10
    n_per = 50
    n = n_schools * n_per
    school_id = np.repeat(np.arange(n_schools), n_per)
    ses = np.random.normal(0, 1, n)
    school_effect = np.random.normal(0, 3, n_schools)[school_id]
    math = 50 + ses - 0.3*ses**2 + school_effect + np.random.normal(0, 2, n)
    return pd.DataFrame({
        "math": math,
        "ses": ses,
        "school_id": school_id
    })


def test_basic_run():
    """Test: model runs and returns ICC."""
    df = make_test_data()
    model = MultiSpline(
        data=df,
        outcome="math",
        predictor="ses",
        cluster="school_id",
        nknots=4
    )
    model.fit()
    assert model.icc_ is not None
    assert 0 <= model.icc_ <= 1
    print(f"  test_basic_run: PASSED (ICC={model.icc_:.4f})")
    return model


def test_fitted_shape():
    """Test: fitted values same length as data."""
    df = make_test_data()
    model = MultiSpline(
        data=df,
        outcome="math",
        predictor="ses",
        cluster="school_id",
        nknots=4
    )
    model.fit()
    fitted = model.predict()
    assert len(fitted) == len(df)
    print(f"  test_fitted_shape: PASSED (n={len(fitted)})")


def test_grid_predict():
    """Test: grid prediction returns correct shape."""
    df = make_test_data()
    model = MultiSpline(
        data=df,
        outcome="math",
        predictor="ses",
        cluster="school_id",
        nknots=4
    )
    model.fit()
    grid = model.predict(grid=True, n_grid=50)
    assert grid.shape == (50, 2)
    print(f"  test_grid_predict: PASSED (shape={grid.shape})")


def test_autoknots():
    """Test: autoknots selects valid number of knots."""
    df = make_test_data()
    model = MultiSpline(
        data=df,
        outcome="math",
        predictor="ses",
        cluster="school_id",
        autoknots=True
    )
    model.fit()
    assert 4 <= model.nknots <= 7
    print(f"  test_autoknots: PASSED (nknots={model.nknots})")


if __name__ == "__main__":
    print("=" * 40)
    print("Running MultiSpline Tests")
    print("=" * 40)
    test_basic_run()
    test_fitted_shape()
    test_grid_predict()
    test_autoknots()
    print("=" * 40)
    print("All tests PASSED!")
    print("=" * 40)