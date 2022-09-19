import torch.nn as nn
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.utils import QuantumInstance
from qiskit_machine_learning import neural_networks
from qiskit_machine_learning.connectors import TorchConnector

NUM_QUBITS = 4


def parity(bitstring: str) -> int:
    return sum([int(bit) for bit in str(bitstring)]) % 2


def create_qnn(noisy: bool = False) -> neural_networks.NeuralNetwork:
    backend = Aer.get_backend("aer_simulator")
    shots = 100

    if noisy:
        noise_model = NoiseModel.from_backend(FakeMontreal())
        qi = QuantumInstance(
            backend=backend,
            noise_model=noise_model,
            shots=shots,
        )
    else:
        qi = QuantumInstance(backend=backend, shots=shots)

    fm = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=2)
    ansatz = RealAmplitudes(num_qubits=NUM_QUBITS, reps=1)
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(fm, range(NUM_QUBITS))
    qc.append(ansatz, range(NUM_QUBITS))

    qnn = neural_networks.CircuitQNN(
        qc,
        input_params=fm.parameters,
        weight_params=ansatz.parameters,
        interpret=parity,
        output_shape=2,
        quantum_instance=qi,
    )

    return qnn


class QNN(nn.Module):
    def __init__(self, noisy: bool = False):
        super().__init__()
        self.qnn = TorchConnector(create_qnn(noisy=noisy))

    def forward(self, x):
        x = self.qnn.forward(x)
        return x


if __name__ == "__main__":

    ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=2).decompose().draw(
        output="latex", filename="fm.pdf"
    )

    RealAmplitudes(num_qubits=NUM_QUBITS, reps=1).decompose().draw(
        output="latex", filename="ansatz.pdf"
    )
