# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
n = 10

np.random.seed(1337)
x = np.linspace(-2, 2, n)
y = x**3 - (2 * x) + np.random.normal(0, 0.5, n)

# %% fit polynomials
x_pred = np.linspace(-2, 2, 100)
y_true = x_pred**3 - (2 * x_pred)

preds = {}
preds["true"] = y_true

for deg in [1, 3, 9]:
    p = np.polyfit(x, y, deg)
    y_pred = np.poly1d(p)(x_pred)

    preds[f"deg_{deg}"] = y_pred

    plt.scatter(x, y, label="data")
    plt.plot(x_pred, y_pred, label="degree %d" % deg)
    plt.plot(x_pred, y_true, c="r")
    plt.legend()
    plt.show()

# %% save data
pd.DataFrame({"x": x, "y": y}).to_csv("samples.csv", index=False)
pd.DataFrame(preds, index=x_pred).to_csv("preds.csv", index_label="x")
# %%
