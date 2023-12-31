import numpy as np
import statsmodels.api as sm

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e

model = sm.OLS(y, X)
results = model.fit()

# do either
print(results.summary().as_latex())

# alternatively
for table in results.summary().tables:
    print(table.as_latex_tabular())