import numpy as np
from patsy import dmatrix, build_design_matrices


def make_spline_basis(x, nknots=4, autoknots=False):
    """
    Create natural cubic spline basis WITHOUT intercept.

    Parameters
    ----------
    x         : array-like predictor
    nknots    : int number of knots (default=4)
    autoknots : bool auto-select knots using floor(sqrt(n))

    Returns
    -------
    B      : np.ndarray spline basis
    knots  : list knot locations
    nknots : int actual number of knots used
    design : patsy DesignMatrix info (for grid prediction)
    """
    x = np.asarray(x, dtype=float)

    if autoknots:
        n_unique = len(np.unique(x))
        nknots = int(np.floor(np.sqrt(n_unique)))
        nknots = max(4, min(nknots, 7))
        print(f"  Autoknots selected: {nknots}")

    quantiles = np.linspace(0, 100, nknots + 2)[1:-1]
    knots = np.percentile(x, quantiles)

    knots_str = ", ".join([str(k) for k in knots])
    formula = f"cr(x, knots=[{knots_str}]) - 1"

    design = dmatrix(formula, {"x": x}, return_type="dataframe")
    B = design.to_numpy()

    print(f"Step 1: Knots at: {np.round(knots, 2)}")
    print(f"Step 2: Spline basis created ({B.shape[1]} terms)")

    return B, knots.tolist(), nknots, design.design_info


def make_grid_basis(x_grid, design_info):
    """
    Build spline basis for a new grid using same knots.

    Parameters
    ----------
    x_grid      : np.ndarray grid of x values
    design_info : patsy design info from training

    Returns
    -------
    B_grid : np.ndarray spline basis on grid
    """
    B_grid = build_design_matrices(
        [design_info],
        {"x": x_grid}
    )[0]
    return np.array(B_grid)