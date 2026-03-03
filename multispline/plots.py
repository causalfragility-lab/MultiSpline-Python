import numpy as np
import matplotlib.pyplot as plt


def plot_fit(x, fitted, predictor, outcome, n_grid=100):
    """
    Plot smooth predicted curve on a grid (publication-ready).

    Parameters
    ----------
    x         : array predictor values
    fitted    : array fitted values
    predictor : str predictor name
    outcome   : str outcome name
    n_grid    : int grid resolution (default=100)
    """
    x = np.array(x, dtype=float)
    fitted = np.array(fitted, dtype=float)

    # Sort
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = fitted[sort_idx]

    # Grid-based smooth curve
    x_grid = np.linspace(x_sorted.min(), x_sorted.max(), n_grid)
    y_grid = np.interp(x_grid, x_sorted, y_sorted)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_grid, y_grid, color="#1a5276", linewidth=2.5)
    ax.set_xlabel(predictor, fontsize=12)
    ax.set_ylabel(f"Predicted {outcome}", fontsize=12)
    ax.set_title(
        "MultiSpline: predicted nonlinear relationship\n"
        "Natural cubic spline | Population-level curve",
        fontsize=12
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("Step 5: Plot generated (grid-based)!")