import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from torch import Tensor, tensor


def load_data() -> tuple[Tensor, Tensor]:
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )

    df = df[df["target"] == 0 | (df["target"] == 1)]
    x = normalize(df.drop("target", axis=1).values)
    y = df["target"].values.astype(int)

    return tensor(x).double(), tensor(y).long()
