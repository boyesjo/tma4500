import pennylane as qml
from pennylane import numpy as np

RG_PARAMS = 3
CONV_PARAMS = 15
POOL_PARAMS = 3
WIRES = 8
TOTAL_PARAMS = 231

dev = qml.device("default.qubit", wires=WIRES)

np.random.seed(1337)


def rg_gate(params, wires):

    assert type(wires) == int
    assert len(params) == RG_PARAMS

    qml.RZ(params[0], wires=wires)
    qml.RY(params[1], wires=wires)
    qml.RZ(params[2], wires=wires)


def conv_gate(params, wires):

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

    high = 0

    for _ in range(3):

        low, high = high, high + layer_size * CONV_PARAMS
        conv_layer(params[low:high], wires=range(0, WIRES, gap))

        low, high = high, high + layer_size * POOL_PARAMS // 2
        pool_layer(params[low:high], wires=range(0, WIRES, gap))

        gap *= 2
        layer_size //= 2

    return qml.expval(qml.PauliZ(4))
