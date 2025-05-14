import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService

# Initialize the service (load credentials from saved account)
service = QiskitRuntimeService()

# Get the least busy real quantum backend
backend = service.least_busy(simulator=False, operational=True)
print(f"Using backend: {backend.name}")

# Define the 3-qubit similarity circuit
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

# Set similarity parameter (0.0 or 1.0)
base_similarity = 1.0  # or 0.0
circuit = create_similarity_circuit(base_similarity)

# Transpile for the target backend
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(circuit)

# Define observables (3-qubit Pauli operators)
observables_labels = ["IIZ", "IIX", "IZI", "IXI", "ZIZ", "XIX"]
observables = [SparsePauliOp(label) for label in observables_labels]
mapped_observables = [obs.apply_layout(isa_circuit.layout) for obs in observables]

# Run on real hardware
estimator = Estimator(backend)
estimator.options.resilience_level = 1
estimator.options.default_shots = 10000  # Increase for better accuracy

job = estimator.run([(isa_circuit, mapped_observables)])
print(f"Job submitted! ID: {job.job_id()}")

# Wait for results (may take time depending on queue)
result = job.result()
pub_result = result[0]
print(pub_result.data.__dict__)
print("Expectation values:", pub_result.data.evs)  # Prints measured observables
