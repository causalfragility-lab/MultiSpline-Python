import pandas as pd
import numpy as np
from multispline import MultiSpline

# Example 1: Education data
np.random.seed(42)
n_schools = 20
n_per_school = 50
n = n_schools * n_per_school

school_id = np.repeat(np.arange(n_schools), n_per_school)
ses = np.random.normal(0, 1, n)
school_effect = np.random.normal(0, 3, n_schools)[school_id]
math = 50 + 0.9*ses - 0.25*ses**2 + school_effect + np.random.normal(0, 2, n)

df_sim = pd.DataFrame({
    "math": math,
    "ses": ses,
    "school_id": school_id
})

print("=" * 50)
print("Example 1: Simulated Education Data")
print("=" * 50)

model1 = MultiSpline(
    data=df_sim,
    outcome="math",
    predictor="ses",
    cluster="school_id",
    nknots=4
)
model1.fit()
model1.summary()
model1.plot()

# Example 2: Labor economics (realistic ICC)
np.random.seed(123)
n = 2000
n_industries = 12

industry = np.random.randint(0, n_industries, n)
age = np.random.uniform(34, 46, n)
ind_effect = np.random.normal(0, 1.5, n_industries)[industry]
wage = (8 + 0.1*age - 0.003*age**2 +
        ind_effect +
        np.random.normal(0, 4, n))

df_labor = pd.DataFrame({
    "wage": wage,
    "age": age,
    "industry": industry
})

print("=" * 50)
print("Example 2: Labor Economics Data")
print("=" * 50)

model2 = MultiSpline(
    data=df_labor,
    outcome="wage",
    predictor="age",
    cluster="industry",
    nknots=4
)
model2.fit()
model2.summary()
model2.plot()