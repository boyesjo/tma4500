# %% https://qiskit.org/documentation/machine-learning/tutorials/11_quantum_convolutional_neural_networks.html # noqa
import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeMontreal as FakeBackend
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import (
    NeuralNetworkClassifier,
)
from qiskit_machine_learning.neural_networks import TwoLayerQNN

# set up noise quantuminstance

noise_model = NoiseModel.from_backend(FakeBackend())
simulator = Aer.get_backend("aer_simulator")
qi = QuantumInstance(
    simulator,
    shots=256,
    # noise_model=noise_model,
)


# %%
def generate_data(n: int = 20):
    # generate 3x3 images with either vertical or horizontal lines
    # and labels 0 or 1
    X = np.zeros((n, 4, 4))
    y = np.zeros(n, dtype=int)

    for i in range(n):

        # horizontal line
        if algorithm_globals.random.integers(0, 2):
            col = algorithm_globals.random.integers(0, 3)
            X[i, col] = 1
            y[i] = 1

        # vertical line
        else:
            row = algorithm_globals.random.integers(0, 3)
            X[i, :, row] = 1
            y[i] = -1

    # flatten images
    X = X.reshape(n, -1)

    # add gaussian noise
    X += algorithm_globals.random.normal(0, 0.05, X.shape)
    # rescale X
    X = (X - X.min()) / (X.max() - X.min())
    return X, y


x, y = generate_data(100)

# %%
# show first 6 images
fig, axes = plt.subplots(2, 3)
for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(x[i].reshape(4, 4), cmap="gray")
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth("0")
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig("data.pdf")

# %%
# split into test and train randomly
idx = algorithm_globals.random.permutation(len(x))
train_idx = idx[: int(len(x) * 0.7)]
test_idx = idx[int(len(x) * 0.7) :]
x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]


# %%
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(
            conv_circuit(params[param_index : (param_index + 3)]), [q1, q2]
        )
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(
            conv_circuit(params[param_index : (param_index + 3)]), [q1, q2]
        )
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(
            pool_circuit(params[param_index : (param_index + 3)]),
            [source, sink],
        )
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


# %%
feature_map = ZFeatureMap(feature_dimension=16)
feature_map.decompose().draw(output="mpl")

# %%
ansatz = QuantumCircuit(16)
ansatz.compose(conv_layer(16, "conv1"), range(16), inplace=True)
ansatz.compose(
    pool_layer(list(range(0, 8)), list(range(8, 16)), "pool1"),
    range(16),
    inplace=True,
)
ansatz.compose(conv_layer(8, "conv2"), range(8, 16), inplace=True)
ansatz.compose(
    pool_layer(list(range(0, 4)), list(range(4, 8)), "pool2"),
    range(8, 16),
    inplace=True,
)
ansatz.compose(conv_layer(4, "conv3"), range(12, 16), inplace=True)
ansatz.compose(
    pool_layer(list(range(0, 2)), list(range(2, 4)), "pool3"),
    range(12, 16),
    inplace=True,
)
ansatz.compose(conv_layer(2, "conv4"), range(14, 16), inplace=True)
ansatz.compose(
    pool_layer(list(range(0, 1)), list(range(1, 2)), "pool4"),
    range(14, 16),
    inplace=True,
)
observable = PauliSumOp.from_list([("Z" + "I" * 15, 1)])
circuit = QuantumCircuit(16)
circuit.compose(feature_map, range(16), inplace=True)
circuit.compose(ansatz, range(16), inplace=True)
qnn = TwoLayerQNN(
    16,
    feature_map=feature_map,
    ansatz=ansatz,
    observable=observable,
    exp_val=AerPauliExpectation(),
    quantum_instance=qi,
)

circuit.draw("mpl")


# %%
def callback_graph(_, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    print(f"Objective function value: {obj_func_eval}")


opflow_classifier = NeuralNetworkClassifier(
    qnn,
    optimizer=COBYLA(maxiter=400),
    callback=callback_graph,
    initial_point=algorithm_globals.random.random(qnn.num_weights),
)

x = np.asarray(x_train)
y = np.asarray(y_train)

objective_func_vals: list[float] = []
opflow_classifier.fit(x, y)

# score classifier
print(
    "Accuracy from the train data: "
    f"{np.round(100 * opflow_classifier.score(x, y), 2)}%"
)

# %%
y_predict = opflow_classifier.predict(x_test)
x = np.asarray(x_test)
y = np.asarray(y_test)
print(
    "Accuracy from the test data: "
    f"{np.round(100 * opflow_classifier.score(x, y), 2)}%"
)

# plot 6 test images
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(x_test[i].reshape(3, 3), cmap="gray")
    ax.set_title(
        "predicted horizontal" if y_predict[i] == 1 else "predicted vertical"
    )
    ax.axis("off")

# # %%
# import pandas as pd

# noisy_optim = pd.read_csv("noisy_optim.csv", names=["Noisy"])
# optim = pd.Series(objective_func_vals, name="Pure")
# df = pd.concat([noisy_optim, optim], names=["Noisy", "Pure"], axis=1)
# df.plot()

# # save df to csv
# df.to_csv("optim.csv", index=True)
# # %%

# %%
