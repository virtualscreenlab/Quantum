import matplotlib
import qiskit

from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import circuit_drawer
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from qiskit_ibm_runtime import QiskitRuntimeService

import matplotlib.pyplot as plt
import numpy as np

# Replace these with your actual IBM Quantum credentials
token = "API_token"
instance = "ibm-q/open/main"  # Optional, e.g., "ibm-q/open/main"

# Save the account
QiskitRuntimeService.save_account(channel="ibm_quantum", token=token, instance=instance, overwrite=True)

from qiskit_ibm_runtime import QiskitRuntimeService

# List all saved accounts
print(QiskitRuntimeService.saved_accounts())

from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize the service
service = QiskitRuntimeService()

# Check available backends
backend = service.least_busy(simulator=False, operational=True)
print(backend)

def create_similarity_circuit(base_similarity):
    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(qr, cr)
    
    theta = 2 * np.arccos(np.sqrt(base_similarity)) if base_similarity > 0 else np.pi
    qc.ry(theta, qr[0])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[2])
    
    qc.measure(qr, cr)
    
    return qc

# Example usage
base_similarity = 0.5
circuit = create_similarity_circuit(base_similarity)

# Visualize the circuit in color
circuit_drawer(circuit, output='mpl')


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer
import numpy as np

def create_similarity_circuit(base_similarity):
    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(qr, cr)
    
    theta = 2 * np.arccos(np.sqrt(base_similarity)) if base_similarity > 0 else np.pi
    qc.ry(theta, qr[0])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[2])
    
    qc.measure(qr, cr)
    
    return qc

# Create the circuit
base_similarity = 0.5
circuit = create_similarity_circuit(base_similarity)

service = QiskitRuntimeService()
 
backend = service.least_busy(simulator=False, operational=True)
 
# Convert to an ISA circuit and layout-mapped observables.
qc = circuit
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)
 
isa_circuit.draw("mpl", idle_wires=False)

# Set up six different 3-qubit observables
observables_labels = ["IIZ", "IIX", "IZI", "IXI", "ZIZ", "XIX"]  # Fixed 3-qubit versions
observables = [SparsePauliOp(label) for label in observables_labels]

# Later when applying layout:
mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]

# Construct the Estimator instance.
 
estimator = Estimator(mode=backend)
estimator.options.resilience_level = 1
estimator.options.default_shots = 5000
 
mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]
 
# One pub, with one circuit to run against five different observables.
job = estimator.run([(isa_circuit, mapped_observables)])
 
# Use the job ID to retrieve your job data later
print(f">>> Job ID: {job.job_id()}")

# This is the result of the entire submission.  You submitted one Pub,
# so this contains one inner result (and some metadata of its own).
job_result = job.result()
 
# This is the result from our single pub, which had six observables,
# so contains information on all six.
pub_result = job.result()[0]

# Use the following code instead if you want to run on a simulator:
 
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
backend = FakeAlmadenV2()
estimator = Estimator(backend)
 
# Convert to an ISA circuit and layout-mapped observables.
 
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)
mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]
 
job = estimator.run([(isa_circuit, mapped_observables)])
result = job.result()
 
# This is the result of the entire submission.  You submitted one Pub,
# so this contains one inner result (and some metadata of its own).
 
job_result = job.result()
 
# This is the result from our single pub, which had five observables,
# so contains information on all five.
 
pub_result = job.result()[0]

print(pub_result.data.__dict__)


observables = ['IIZ', 'IIX', 'IZI', 'IXI', 'ZIZ', 'XIX']
evs = np.array([ 0.99121094,  0.05322266,  0.92578125,  0.01611328,  0.97900391, -0.01464844])
stds = np.array([0.00206705, 0.01560285, 0.00590718, 0.01562297, 0.00318502, 0.01562332])

plt.figure(figsize=(12, 6))
plt.bar(observables, evs, yerr=stds, capsize=5)
plt.title("Expectation Values of Observables")
plt.xlabel("Observables")
plt.ylabel("Expectation Value")
plt.ylim(-1, 1)
plt.axhline(y=0, color='r', linestyle='--')

for i, v in enumerate(evs):
    plt.text(i, v, f'{v:.4f}', ha='center', va='bottom' if v >= 0 else 'top')

plt.tight_layout()
plt.show()
