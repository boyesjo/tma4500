import pandas as pd

df_nn = pd.read_csv("results/test_nn.csv").transpose()
df_qnn = pd.read_csv("results/test_qnn.csv").transpose()
df_qnn_noisy = pd.read_csv("results/test_qnn_noisy.csv").transpose()


# take means and combine into one df
df = pd.DataFrame(
    {
        "NN": df_nn.mean(axis=1),
        "QNN": df_qnn.mean(axis=1),
        "QNN Noisy": df_qnn_noisy.mean(axis=1),
    }
)

# set index as int
df.index = df.index.astype(int)
# sort index
df = df.sort_index()

# save
df.to_csv("results/mean.csv", na_rep="nan", index_label="Index")
