# %%
import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.algorithms.optimizers import ADAM
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import (
    NeuralNetworkClassifier,
)
from qiskit_machine_learning.neural_networks import TwoLayerQNN

algorithm_globals.random_seed = 1337

simulator = Aer.get_backend("aer_simulator")
qi = QuantumInstance(
    simulator,
    shots=256,
)


# %%
def rg_gate(params):
    if len(params) != 3:
        raise ValueError("params must be of length 3")
    qc = QuantumCircuit(1, name="$R_G$")
    qc.rz(params[0], 0)
    qc.ry(params[1], 0)
    qc.rz(params[2], 0)
    return qc.to_gate()


def w_gate(params: np.ndarray):
    assert len(params) == 15
    qc = QuantumCircuit(2, name="$W$")

    qc.append(rg_gate(params[:3]), [0])
    qc.append(rg_gate(params[3:6]), [1])

    qc.cx(1, 0)

    qc.rz(params[6], 0)
    qc.ry(params[7], 1)

    qc.cx(0, 1)

    qc.ry(params[8], 1)

    qc.cx(1, 0)

    qc.append(rg_gate(params[9:12]), [0])
    qc.append(rg_gate(params[12:15]), [1])

    return qc.to_gate()


qc = QuantumCircuit(2)
qc.append(w_gate(np.zeros(15)), [0, 1])
qc.decompose().draw(output="mpl")


# %%
def i_gate():
    q = QuantumRegister(2)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c, name="$I$")
    qc.measure(q[0], c)
    qc.x(1).c_if(c, 1)
    return qc.to_instruction()


# %%
def conv_layer(num_qubits, param_prefix):
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    params = ParameterVector(param_prefix, 15 * (num_qubits - 1))

    param_index = 0

    for i in range(1, num_qubits - 1, 2):
        qc = qc.compose(
            w_gate(params[param_index : param_index + 15]), [i, i + 1]
        )
        qc.barrier()
        param_index += 15

    for i in range(0, num_qubits - 1, 2):
        qc = qc.compose(
            w_gate(params[param_index : param_index + 15]), [i, i + 1]
        )
        qc.barrier()
        param_index += 15

    # qc_inst = qc.to_instruction()
    # qc = QuantumCircuit(num_qubits, 1, name="Conv Layer")
    # qc.append(qc_inst, range(num_qubits))
    return qc


# plot the conv layer
conv_layer(4, "_").draw(output="mpl")


# %%
def pool_layer(num_qubits):
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # measure first qubit in upper half, second in lower
    middle = 2 * (num_qubits // 4)

    for i in range(0, middle, 2):
        qc = qc.compose(i_gate(), [i, i + 1])
        qc.barrier()

    for i in range(middle, num_qubits - 1, 2):
        qc = qc.compose(i_gate(), [i + 1, i])
        qc.barrier()

    return qc


pool_layer(8).draw(output="mpl")


# %%
def create_ansatz():
    num_qubits = 8
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr, name="Ansatz")

    # add conv layer
    qc = qc.compose(conv_layer(num_qubits, "conv"), range(num_qubits))
    qc.barrier()

    # add pool layer
    qc = qc.compose(pool_layer(num_qubits), range(num_qubits))
    qc.barrier()

    active_qubits = list(range(2, 6))

    # add conv layer
    qc = qc.compose(conv_layer(len(active_qubits), "conv2"), active_qubits)
    qc.barrier()

    # add pool layer
    qc = qc.compose(pool_layer(len(active_qubits)), active_qubits)
    qc.barrier()

    active_qubits = list(range(3, 5))

    # add dense layer (RealAmplitudes)
    qc = qc.compose(RealAmplitudes(len(active_qubits)), active_qubits)

    return qc


create_ansatz().draw(output="mpl")


# %%
qnn = TwoLayerQNN(
    num_qubits=8,
    feature_map=ZFeatureMap(8),
    ansatz=create_ansatz(),
    quantum_instance=qi,
    exp_val=AerPauliExpectation(),
    observable=PauliSumOp.from_list([("IIIZZIII", 1)]),
)

qnn.circuit.draw(output="mpl")


# %%
def generate_data(n: int = 32):
    x = np.zeros((2 * n, 4, 2))
    y = np.zeros(2 * n, dtype=int)

    for i in range(n):
        x[i, i % 4] = 1
        y[i] = 1

    for i in range(n):
        x[n + i][:, i % 2] = 1
        y[n + i] = -1

    # flatten images
    x = x.reshape(2 * n, 8)
    # rescale to (0, pi/2)
    x *= np.pi / 2

    # add gaussian noise
    x += algorithm_globals.random.normal(0, 0.1, x.shape)

    # shuffle data
    idx = np.arange(2 * n)
    algorithm_globals.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y


x_train, y_train = generate_data(32)
x_test, y_test = generate_data(8)


# %%
def score(weights, x, y):
    y_pred = qnn.forward(x, weights=weights)
    y_pred = np.where(y_pred > 0, 1, -1).flatten()
    return np.mean(y_pred == y)


def callback(weights, loss):
    stats = {
        "iteration": len(history),
        "loss": loss,
        "training_acc": score(weights, x_train, y_train),
        "test_acc": score(weights, x_test, y_test),
    }
    # print stats with 2 decimal places
    print(
        f"iteration: {stats['iteration']:4}, "
        f"loss: {stats['loss']:4.3f}, "
        f"training_acc: {stats['training_acc']:3.2f}, "
        f"test_acc: {stats['test_acc']:3.2f}"
    )
    history.append(stats)


init_weights = algorithm_globals.random.random(qnn.num_weights)

opflow_classifier = NeuralNetworkClassifier(
    qnn,
    loss="absolute_error",
    optimizer=ADAM(maxiter=400),
    callback=callback,
    initial_point=init_weights,
)

history = []
opflow_classifier.fit(x_train, y_train)

# %%
