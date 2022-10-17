# %%
import pennylane as qml
from gen_data import generate_data
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

CONV_PARAMS = 3
POOL_PARAMS = 3
WIRES = 8
TOTAL_PARAMS = 90

dev = qml.device("default.qubit", wires=WIRES)

np.random.seed(1337)


# %%
def conv_gate(params, wires):
    assert len(wires) >= 2
    assert len(params) == CONV_PARAMS

    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi / 2, wires=wires[0])


def conv_layer(params, wires):
    assert len(wires) >= 2
    assert len(params) == CONV_PARAMS * (len(wires))

    param_index = 0

    for i in range(len(wires)):
        conv_gate(
            params[param_index : param_index + CONV_PARAMS],
            wires=[wires[i], wires[(i + 1) % len(wires)]],
        )
        param_index += CONV_PARAMS


def pool_gate(params, wires):
    assert len(wires) >= 2
    assert len(params) == POOL_PARAMS

    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])


def pool_layer(params, wires):
    assert len(wires) >= 2

    mid = len(wires) // 2

    assert 2 * mid == len(wires)
    assert len(params) == POOL_PARAMS * (len(wires) // 2)

    param_index = 0

    for i in range(0, mid):
        pool_gate(
            params[param_index : param_index + POOL_PARAMS],
            wires=[wires[i], wires[i + mid]],
        )
        param_index += POOL_PARAMS


@qml.qnode(dev)
def circuit(params, x):
    qml.templates.AngleEmbedding(x, wires=range(8))

    layer_size = WIRES
    gap = 1
    for _ in range(3):
        low, high = 0, layer_size * CONV_PARAMS
        conv_layer(params[low:high], wires=range(0, WIRES, gap))

        low, high = high, high + layer_size * POOL_PARAMS // 2
        pool_layer(params[low:high], wires=range(0, WIRES, gap))

        gap *= 2
        layer_size //= 2

    return qml.expval(qml.PauliZ(4))


# # %%
# init_params = np.random.rand(90)
# x = np.random.rand(8)
# circuit(init_params, x)

# # %%
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
