# %%
import numpy as np
import pandas as pd
from load_data import load_data
from nn import NN, LogReg
from qnn import QNN
from sklearn.model_selection import train_test_split
from torch import Tensor, nn, optim

# %%
x, y = load_data()
idx = list(np.random.permutation(len(x)))
x, y = x[idx], y[idx]


# %%
def train(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    epochs: int = 100,
) -> pd.DataFrame:

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.1,
        betas=(0.9, 0.99),
        eps=1e-10,
    )

    loss_func = nn.CrossEntropyLoss()
    history = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

        y_pred = model(x).argmax(dim=1)

        history.append(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "train_acc": (y_pred == y).sum().item() / len(y),
                "test_acc": (model(x_test).argmax(dim=1) == y_test)
                .sum()
                .item(),
            }
        )
        print(history[-1])

    return pd.DataFrame(history)


# %%
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# %%
models = {
    "NN": NN(),
    "LogReg": LogReg(),
    "QNN_last": QNN(interpret="last_bit"),
    "QNN_parity": QNN(interpret="parity"),
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    history = train(model, train_x, train_y, test_x, test_y)
    results[name] = history
    history.to_csv(f"results_5/{name}.csv", index=False)

# %%
import matplotlib.pyplot as plt

for name, history in results.items():
    plt.plot(history["epoch"], history["loss"], label=name)

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# %%
