# MultiSpline for Python

Nonlinear multilevel spline modeling for Python.

[![Version](https://img.shields.io/badge/version-0.1.4-blue.svg)](https://github.com/causalfragility-lab/MultiSpline-Python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

**Version:** 0.1.4
**License:** MIT
**PyPI:** https://pypi.org/project/multispline-python/

---

## Motivation

Nonlinear relationships are common in applied research, especially
in education, health, and economics. While Python provides
statsmodels for mixed-effects models and patsy for spline
construction, combining these tools requires multiple steps and
manual postestimation. multispline addresses this gap by providing
a unified workflow that integrates spline construction, multilevel
estimation, ICC computation, prediction, and visualization into a
single interface, improving workflow efficiency and reproducibility.

---

## Installation

### PyPI (recommended):
```bash
pip install multispline-python
```

### Development version (GitHub):
```bash
pip install git+https://github.com/causalfragility-lab/MultiSpline-Python.git
```

### Dependencies:
```bash
pip install numpy pandas scipy statsmodels patsy matplotlib
```

---

## Quickstart
```python
from multispline import MultiSpline
import pandas as pd

model = MultiSpline(
    data=df,
    outcome="math",
    predictor="ses",
    cluster="school_id",
    nknots=4
)
model.fit()
model.summary()
model.plot()
```

---

## Examples

### Example 1: Education - SES and math achievement
```python
import numpy as np
import pandas as pd
from multispline import MultiSpline

np.random.seed(42)
n_schools = 20
n_per_school = 50
n = n_schools * n_per_school

school_id = np.repeat(np.arange(n_schools), n_per_school)
ses = np.random.normal(0, 1, n)
school_effect = np.random.normal(0, 3, n_schools)[school_id]
math = 50 + 0.9*ses - 0.25*ses**2 + school_effect + np.random.normal(0, 2, n)

df = pd.DataFrame({
    "math": math,
    "ses": ses,
    "school_id": school_id
})

model = MultiSpline(
    data=df,
    outcome="math",
    predictor="ses",
    cluster="school_id",
    nknots=4
)
model.fit()
model.summary()
model.plot()
```

### Example 2: Labor economics - age and wage
```python
np.random.seed(123)
n = 2000
n_industries = 12

industry = np.random.randint(0, n_industries, n)
age = np.random.uniform(34, 46, n)
ind_effect = np.random.normal(0, 1.5, n_industries)[industry]
wage = (8 + 0.1*age - 0.003*age**2 +
        ind_effect + np.random.normal(0, 4, n))

df = pd.DataFrame({
    "wage": wage,
    "age": age,
    "industry": industry
})

model = MultiSpline(
    data=df,
    outcome="wage",
    predictor="age",
    cluster="industry",
    nknots=4,
    autoknots=False
)
model.fit()
model.summary()
model.plot()

# Grid prediction
grid = model.predict(grid=True, n_grid=100)
print(grid.head())
```

---

## Workflow

multispline automates five steps in a single interface:

1. Knot placement at quantiles of predictor distribution
2. Natural cubic spline basis construction via patsy
3. Multilevel model estimation via statsmodels MixedLM
4. ICC computation from variance components
5. Grid-based prediction and visualization

Optimization uses robust fallback: lbfgs -> bfgs -> cg.

---

## Options

| Parameter | Description |
|-----------|-------------|
| `outcome` | Outcome variable name (str) |
| `predictor` | Predictor variable name (str) |
| `cluster` | Grouping variable name (str) |
| `nknots` | Number of spline knots (default=4) |
| `autoknots` | Auto-select knots 4-7 via floor(sqrt(n)) |

## Methods

| Method | Description |
|--------|-------------|
| `.fit()` | Fit the model |
| `.summary()` | Print model summary and ICC |
| `.predict()` | Return fitted values |
| `.predict(grid=True)` | Return grid DataFrame |
| `.plot()` | Visualize nonlinear fit |

---

## Requirements

- Python 3.8 or later
- Predictor variable must be continuous
- Sufficient between-cluster variability required

---

## Limitations

- Predictor variable must be continuous
- Performance depends on sufficient between-cluster variability
- Not appropriate for discrete predictors or negligible random-effects variance
- Random-intercept only (no random slopes in v0.1.4)
- Continuous outcomes only

---

## Related Packages

| Language | Package | Repository |
|----------|---------|------------|
| R | MultiSpline | https://github.com/causalfragility-lab/MultiSpline |
| Stata | multispline | https://github.com/causalfragility-lab/MultiSpline-Stata |
| Python | MultiSpline-Python | this repo |

---

## Author

Subir Hait
Michigan State University
haitsubi@msu.edu
RePEC: https://authors.repec.org/pro/pha1643
GitHub: https://github.com/causalfragility-lab

---

## License

MIT (c) Subir Hait

---

## Citation

If you use multispline in your research, please cite:
```
Hait, Subir. 2026. MultiSpline for Python: Nonlinear multilevel
spline modeling. Version 0.1.4.
https://github.com/causalfragility-lab/MultiSpline-Python
```

### BibTeX:
```bibtex
@software{hait2026multisplinepy,
  author  = {Hait, Subir},
  title   = {MultiSpline for Python: Nonlinear multilevel
             spline modeling},
  year    = {2026},
  version = {0.1.4},
  url     = {https://github.com/causalfragility-lab/MultiSpline-Python}
}
```