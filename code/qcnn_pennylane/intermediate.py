# %%
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
from qiskit.providers import fake_provider
from qiskit.providers.aer.noise import NoiseModel

TOTAL_PARAMS = 10 * 15 + 2 * 2

dev = qml.device(
    "default.qubit",
    # "qiskit.aer",
    # "lightning.qubit",
    wires=8,
    # noise_model=NoiseModel.from_backend(fake_provider.FakeMontreal())
    # shots=1000,
)

np.random.seed(1337)


# %%
def rg_gate(params, wires):
    qml.RZ(params[0], wires=wires)
    qml.RY(params[1], wires=wires)
    qml.RZ(params[2], wires=wires)


def w_gate(params, wires):
    assert len(wires) == 2
    assert len(params) == 15

    rg_gate(params[:3], wires=wires[0])
    rg_gate(params[3:6], wires=wires[1])

    qml.CNOT(wires=[wires[1], wires[0]])

    qml.RZ(params[6], wires=wires[0])
    qml.RY(params[7], wires=wires[1])

    qml.CNOT(wires=[wires[0], wires[1]])

    qml.RY(params[8], wires=wires[1])

    qml.CNOT(wires=[wires[1], wires[0]])

    rg_gate(params[9:12], wires=wires[0])
    rg_gate(params[12:15], wires=wires[1])


def i_gate(wires):
    assert len(wires) == 2
    m = qml.measure(wires=wires[1])
    qml.cond(m, qml.X)(wires=wires[0])


def conv_layer(params, wires):

    assert len(wires) >= 2
    assert len(params) == 15 * (len(wires) - 1)

    param_index = 0

    for i in range(1, len(wires) - 1, 2):
        w_gate(
            params[param_index : param_index + 15],
            wires=[wires[i], wires[i + 1]],
        )
        param_index += 15

    for i in range(0, len(wires) - 1, 2):
        w_gate(
            params[param_index : param_index + 15],
            wires=[wires[i], wires[i + 1]],
        )
        param_index += 15


def pool_layer(wires):

    for i in range(0, len(wires) - 1, 2):
        i_gate(wires=[wires[i], wires[i + 1]])


@qml.qnode(dev)
def circuit(params, x):
    qml.templates.AngleEmbedding(x, wires=range(8))

    conv_layer(params[: 7 * 15], wires=range(8))
    pool_layer(wires=range(8))

    conv_layer(params[7 * 15 : 10 * 15], wires=range(0, 8, 2))
    pool_layer(wires=range(0, 8, 2))

    qml.BasicEntanglerLayers(
        params[10 * 15 :].reshape(2, 2), wires=range(0, 8, 4)
    )

    return qml.expval(qml.PauliZ(0))
    # return [qml.expval(qml.PauliZ(wires=i)) for i in range(0, 8, 4)]


# # %%
# init_params = np.random.rand(10 * 15 + 2 * 2)
# x = np.random.rand(8)
# circuit(init_params, x)


# # %%
# def generate_data(n: int = 32):
#     x = np.zeros((2 * n, 4, 2))
#     y = np.zeros(2 * n, dtype=int)

#     for i in range(n):
#         x[i, i % 4] = 1
#         y[i] = 1

#     for i in range(n):
#         x[n + i][:, i % 2] = 1
#         y[n + i] = -1

#     # flatten images
#     x = x.reshape(2 * n, 8)
#     # rescale to (0, pi/2)
#     x *= np.pi / 2

#     # add gaussian noise
#     x += np.random.normal(0, 0.1, x.shape)

#     # shuffle data
#     idx = np.arange(2 * n)
#     np.random.shuffle(idx)
#     x = x[idx]
#     y = y[idx]

#     return x, y


# x_train, y_train = generate_data(64)
# x_test, y_test = generate_data(64)

# # %%
# opt = AdamOptimizer(0.01)


# def square_loss(labels, predictions):
#     loss = 0
#     for label, prediction in zip(labels, predictions):
#         loss = loss + (label - prediction) ** 2

#     loss = loss / len(labels)
#     return loss


# def cost(var, features, labels):
#     preds = [circuit(var, x) for x in features]
#     return square_loss(labels, preds)


# def accuracy(var, features, labels):
#     preds = [np.sign(circuit(var, x)) for x in features]
#     return np.mean(preds == labels)


# history = []

# params = init_params
# for i in range(100):
#     (params, _, _), loss = opt.step_and_cost(cost, params, x_train, y_train)
#     test_loss = cost(params, x_test, y_test)
#     history.append(
#         {
#             "Loss": float(loss),
#             "Test loss": float(test_loss),
#             "Accuracy": float(accuracy(params, x_train, y_train)),
#             "Test accuracy": float(accuracy(params, x_test, y_test)),
#         }
#     )
#     print(history[-1])


# # %%
# # predict test set
# preds = [circuit(params, x) for x in x_test]
# print(preds)
# print(y_test)
# # %%
# cost(params, x_test, y_test)
# # %%
# pd.DataFrame(
#     history,
#     index=range(1, len(history) + 1),
# ).to_csv("results.csv", index_label="Iteration")


# # %%
