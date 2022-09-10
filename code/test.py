import matplotlib.pyplot as plt
import qiskit as qk
from qiskit.visualization import plot_histogram

# latex rendering
plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters test test
    }
)

qc = qk.QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

backend = qk.Aer.get_backend("qasm_simulator")
job = qk.execute(qc, backend, shots=1000)

plot_histogram(job.result().get_counts())
plt.show()
