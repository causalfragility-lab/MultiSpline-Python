import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from scipy.linalg import qr


def make_full_rank(X, tol=1e-10):
    """Return full-rank version of X by QR with pivoting."""
    X = np.asarray(X, dtype=float)
    Q, R, piv = qr(X, pivoting=True)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return X, np.arange(X.shape[1], dtype=int)
    r = int(np.sum(diag > tol * diag.max()))
    keep = np.sort(piv[:r])
    return X[:, keep], keep


def fit_mixed_model(data, outcome, X_spline, cluster):
    """
    Fit linear mixed-effects model with spline basis.
    Uses robust optimization fallback: lbfgs -> bfgs -> cg.

    Parameters
    ----------
    data     : pd.DataFrame
    outcome  : str outcome variable name
    X_spline : np.ndarray spline basis matrix
    cluster  : str grouping variable

    Returns
    -------
    model      : fitted MixedLM
    fitted     : predicted values
    icc        : intraclass correlation
    fe_params  : fixed effects parameters
    X_full     : full design matrix (for grid prediction)
    mu         : standardization mean
    sd         : standardization SD
    keep_idx   : kept column indices
    """
    df = data.copy()

    # Add intercept ONCE
    X = np.asarray(X_spline, float)
    X = sm.add_constant(X, has_constant="add")

    # Standardize non-constant columns
    X_const = X[:, [0]]
    X_rest  = X[:, 1:]
    mu = X_rest.mean(axis=0)
    sd = X_rest.std(axis=0)
    sd[sd == 0] = 1.0
    X_rest = (X_rest - mu) / sd
    X = np.column_stack([X_const, X_rest])

    # Drop redundant columns
    X, keep_idx = make_full_rank(X, tol=1e-10)

    # Add to dataframe
    spline_cols = []
    for i in range(X.shape[1]):
        colname = f"sp{i}"
        df[colname] = X[:, i]
        spline_cols.append(colname)

    predictors = " + ".join(spline_cols[1:])
    formula = outcome + " ~ " + predictors

    # Robust optimization fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            formula,
            df,
            groups=df[cluster]
        ).fit(
            method=["lbfgs", "bfgs", "cg"],
            reml=True
        )

    var_re    = float(model.cov_re.iloc[0, 0])
    var_resid = float(model.scale)

    if var_re < 0:
        var_re = 0.0

    icc = var_re / (var_re + var_resid) if (var_re + var_resid) > 0 else 0.0

    print("Step 3: Multilevel model fitted! (robust optimizer)")
    print(f"\nStep 4: ICC ({cluster}): {icc:.4f}")
    print(f"        Var(cluster):   {var_re:.4f}")
    print(f"        Var(residual):  {var_resid:.4f}")

    # Warning for negligible clustering
    if icc < 0.01 or var_re < 1e-6:
        print(f"\n  WARNING: Between-cluster variance is negligible.")
        print(f"  ICC ~ 0 suggests random intercept may be unnecessary.")
        print(f"  Consider a single-level model instead.")

    try:
        fitted = model.fittedvalues
    except Exception:
        fitted = model.predict(df)

    fe_params = model.fe_params

    return model, fitted, icc, fe_params, X, mu, sd, keep_idx