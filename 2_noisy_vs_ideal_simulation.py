from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

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

base_similarity = 0.5
circuit = create_similarity_circuit(base_similarity)

# Create a noise model with depolarizing error on CX gates
noise_model = NoiseModel()
error = 0.02  # Depolarizing error probability
noise_model.add_all_qubit_quantum_error(depolarizing_error(error, 2), ['cx'])

# Create noisy and ideal simulators
simulator_noisy = AerSimulator(noise_model=noise_model)
simulator_ideal = AerSimulator()

# Transpile circuits for noisy basis gates
circuit_noisy = transpile(circuit, simulator_noisy)
circuit_ideal = transpile(circuit, simulator_ideal)

# Run simulations
job_noisy = simulator_noisy.run(circuit_noisy, shots=1000)
job_ideal = simulator_ideal.run(circuit_ideal, shots=1000)

# Get results
result_noisy = job_noisy.result()
result_ideal = job_ideal.result()

counts_noisy = result_noisy.get_counts(circuit_noisy)
counts_ideal = result_ideal.get_counts(circuit_ideal)

print("Noisy Counts:", counts_noisy)
print("Ideal Counts:", counts_ideal)
