import numpy as np
import pandas as pd
from .splines import make_spline_basis, make_grid_basis
from .models import fit_mixed_model
from .plots import plot_fit


class MultiSpline:
    """
    Nonlinear multilevel spline modeling.

    Parameters
    ----------
    data      : pd.DataFrame
    outcome   : str outcome variable
    predictor : str predictor variable
    cluster   : str grouping variable
    nknots    : int number of knots (default=4)
    autoknots : bool auto-select knots (default=False)

    Example
    -------
    >>> model = MultiSpline(
    ...     data=df,
    ...     outcome="math",
    ...     predictor="ses",
    ...     cluster="school_id",
    ...     nknots=4
    ... )
    >>> model.fit()
    >>> model.summary()
    >>> model.plot()
    """

    def __init__(self, data, outcome, predictor,
                 cluster, nknots=4, autoknots=False):
        self.data        = data
        self.outcome     = outcome
        self.predictor   = predictor
        self.cluster     = cluster
        self.nknots      = nknots
        self.autoknots   = autoknots
        self.model_      = None
        self.knots_      = None
        self.fitted_     = None
        self.icc_        = None
        self.fe_params_  = None
        self.design_info_= None
        self.mu_         = None
        self.sd_         = None
        self.keep_idx_   = None

    def fit(self):
        """Fit the nonlinear multilevel spline model."""

        print("--------------------------------------------")
        print("  MultiSpline: Nonlinear Multilevel Modeling")
        print("  Version 0.1.0 | Subir Hait | MSU 2026")
        print("--------------------------------------------")
        print(f"  Outcome   : {self.outcome}")
        print(f"  Predictor : {self.predictor}")
        print(f"  Cluster   : {self.cluster}")
        print(f"  Knots     : {'auto' if self.autoknots else self.nknots}")
        print("--------------------------------------------")

        # Steps 1 & 2: Spline basis
        X_spline, self.knots_, self.nknots, self.design_info_ = make_spline_basis(
            self.data[self.predictor],
            nknots=self.nknots,
            autoknots=self.autoknots
        )

        # Steps 3 & 4: Mixed model + ICC
        (self.model_,
         self.fitted_,
         self.icc_,
         self.fe_params_,
         self.X_full_,
         self.mu_,
         self.sd_,
         self.keep_idx_) = fit_mixed_model(
            self.data,
            self.outcome,
            X_spline,
            self.cluster
        )

        print("--------------------------------------------")
        print("  MultiSpline completed successfully!")
        print("--------------------------------------------")

        return self

    def summary(self):
        """Print model summary."""
        if self.model_ is None:
            raise ValueError("Run fit() first!")
        print(self.model_.summary())
        print(f"\nICC ({self.cluster}): {self.icc_:.4f}")
        return self

    def predict(self, grid=False, n_grid=100):
        """
        Return fitted values or grid predictions.

        Parameters
        ----------
        grid   : bool return smooth grid predictions
        n_grid : int number of grid points (default=100)

        Returns
        -------
        If grid=False: pd.Series of fitted values
        If grid=True:  pd.DataFrame with x_grid, y_grid
        """
        if self.model_ is None:
            raise ValueError("Run fit() first!")

        if grid:
            x = np.array(self.data[self.predictor], dtype=float)
            x_grid = np.linspace(x.min(), x.max(), n_grid)

            # Build basis on grid using same knots
            B_grid = make_grid_basis(x_grid, self.design_info_)

            # Apply same standardization
            import statsmodels.api as sm
            X_grid = sm.add_constant(B_grid, has_constant="add")
            X_grid_rest = (X_grid[:, 1:] - self.mu_) / self.sd_
            X_grid = np.column_stack([X_grid[:, [0]], X_grid_rest])
            X_grid = X_grid[:, self.keep_idx_]

            # Fixed effects prediction (population-level)
            y_grid = X_grid @ self.fe_params_.values

            return pd.DataFrame({
                self.predictor: x_grid,
                f"predicted_{self.outcome}": y_grid
            })

        return self.fitted_

    def plot(self):
        """Plot predicted nonlinear relationship (grid-based)."""
        if self.fitted_ is None:
            raise ValueError("Run fit() first!")
        plot_fit(
            self.data[self.predictor],
            self.fitted_,
            self.predictor,
            self.outcome
        )
        return self