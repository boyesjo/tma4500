import numpy as np
import pandas as pd
from load_data import load_data
from nn import NN
from qnn import QNN
from torch import Tensor, nn, optim

np.random.seed(0)
x, y = load_data()
idx = list(np.random.permutation(len(x)))
x, y = x[idx], y[idx]


def train(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    epochs: int = 100,
) -> np.ndarray:

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.1,
        betas=(0.9, 0.99),
        eps=1e-10,
    )
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.NLLLoss()
    test_list = np.zeros(epochs)  # Store loss history

    model.train()  # Set model to training mode
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)  # Initialise gradient
        output = model(x)  # Forward pass
        loss = loss_func(output, y)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights

        y_pred = model(x).argmax(dim=1)
        acc = (y_pred == y).sum().item() / len(y)
        test_list[epoch] = (
            model(x_test).argmax(dim=1) == y_test
        ).sum().item() / len(y_test)

        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Loss: {loss.item():.4f}, "
            f"Train accuracy: {acc:.2f}, "
            f"Test accuracy: {test_list[epoch]:.2f}"
        )

    return test_list


k = 10
epochs = 100
fold_len = len(x) // k
test_nn = np.zeros((k, epochs))
test_qnn = np.zeros((k, epochs))
for i in range(k):
    # take k-th fold
    test_idx = list(range(i * fold_len, (i + 1) * fold_len))
    train_idx = list(range(0, i * fold_len)) + list(
        range((i + 1) * fold_len, k * fold_len)
    )

    print(test_idx)

    test_nn[i] = train(
        NN(),
        x[train_idx],
        y[train_idx],
        x[test_idx],
        y[test_idx],
        epochs=epochs,
    )

    # test_qnn[i] = train(
    #     QNN(),
    #     x[train_idx],
    #     y[train_idx],
    #     x[test_idx],
    #     y[test_idx],
    #     epochs=epochs,
    # )

# save as csv using pandas
df_nn = pd.DataFrame(test_nn)
df_nn.to_csv("test_nn2.csv", index=False)
# df_qnn = pd.DataFrame(test_qnn)
# df_qnn.to_csv("test_qnn.csv", index=False)


# model_nn = NN()
# model_qnn = QNN()

# loss_list_nn = train(model_nn, epochs=100)
# # save loss with time stamp
# pd.DataFrame(loss_list_nn).to_csv(f"code/iris/nn_loss/{time.time()}.csv")

# loss_list_qnn = train(model_qnn, epochs=100)
# pd.DataFrame(loss_list_qnn).to_csv(f"code/iris/qnn_loss/{time.time()}.csv")

# # plot both loss curves
# plt.plot(loss_list_nn, label="NN", color="blue")
# plt.plot(loss_list_qnn, label="QNN", color="red")
# plt.legend()
# plt.show()
