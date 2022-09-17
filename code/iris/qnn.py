import torch
import torch.nn as nn
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning import neural_networks
from qiskit_machine_learning.connectors import TorchConnector

NUM_QUBITS = 4

qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector"), shots=100)
fm = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=2)
ansatz = RealAmplitudes(num_qubits=NUM_QUBITS, reps=1)
qc = QuantumCircuit(4)
qc.append(fm, range(4))
qc.append(ansatz, range(4))


def parity(bitstring: str) -> int:
    return sum([int(bit) for bit in str(bitstring)]) % 2


# qnn = neural_networks.TwoLayerQNN(
#     4,
#     feature_map=fm,
#     ansatz=ansatz,
#     quantum_instance=qi,
# )

qnn = neural_networks.CircuitQNN(
    qc,
    input_params=fm.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=2,
    quantum_instance=qi,
)


class QNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn = TorchConnector(qnn)

    def forward(self, x):
        x = self.qnn.forward(x)
        return x
        # x = (x + 1) / 2
        # return torch.cat((x, 1 - x), -1)


if __name__ == "__main__":
    import numpy as np
    from load_data import load_data

    x, y = load_data()
    print(qnn.forward(x, weights=np.zeros(8)))
    print(QNN()(x))
