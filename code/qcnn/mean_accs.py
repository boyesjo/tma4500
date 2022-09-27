# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_mean_accs(filename, window=100):
    df = pd.read_csv(filename, index_col="iteration")

    # get running mean and std
    df["test_mean"] = df["test_acc"].rolling(window, min_periods=1).mean()
    df["test_std"] = df["test_acc"].rolling(window, min_periods=1).std()

    df["train_mean"] = df["training_acc"].rolling(window, min_periods=1).mean()
    df["train_std"] = df["training_acc"].rolling(window, min_periods=1).std()

    return df[["test_mean", "test_std", "train_mean", "train_std"]]


# %%
df_noisy = get_mean_accs("noisy.csv")
df_exact = get_mean_accs("exact.csv")

# %%
# plot nosiy and exact with training and test
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(df_noisy["test_mean"], label="noisy test")
# ax.fill_between(
#     df_noisy.index,
#     df_noisy["test_mean"] - df_noisy["test_std"],
#     df_noisy["test_mean"] + df_noisy["test_std"],
#     alpha=0.2,
# )
ax.plot(df_noisy["train_mean"], label="noisy train")
# ax.fill_between(
#     df_noisy.index,
#     df_noisy["train_mean"] - df_noisy["train_std"],
#     df_noisy["train_mean"] + df_noisy["train_std"],
#     alpha=0.2,
# )
ax.plot(df_exact["test_mean"], label="exact test")
# ax.fill_between(
#     df_exact.index,
#     df_exact["test_mean"] - df_exact["test_std"],
#     df_exact["test_mean"] + df_exact["test_std"],
#     alpha=0.2,
# )
ax.plot(df_exact["train_mean"], label="exact train")
# ax.fill_between(
#     df_exact.index,
#     df_exact["train_mean"] - df_exact["train_std"],
#     df_exact["train_mean"] + df_exact["train_std"],
#     alpha=0.2,
# )
ax.set_xlabel("Iteration")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy of noisy and exact training")
ax.legend()
plt.show()


# %%
df = df_noisy.join(df_exact, lsuffix="_noisy", rsuffix="_exact")

# drop stds
df = df.drop(
    columns=[
        "test_std_noisy",
        "train_std_noisy",
        "test_std_exact",
        "train_std_exact",
    ]
)

df.to_csv("mean_accs.csv")

# %%
