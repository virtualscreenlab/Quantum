from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
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

# Create circuit
circuit = create_similarity_circuit(0.5)

# Simulate ideal results
ideal_simulator = Aer.get_backend('qasm_simulator')
ideal_counts = ideal_simulator.run(circuit, shots=10000).result().get_counts()

# Simulate noisy results
fake_backend = FakeManilaV2()
noisy_simulator = AerSimulator.from_backend(fake_backend)
transpiled_circuit = transpile(circuit, fake_backend)
noisy_counts = noisy_simulator.run(transpiled_circuit, shots=10000).result().get_counts()

# Plot noise probability
noise_prob = {k: abs(ideal_counts.get(k, 0)/10000 - noisy_counts.get(k, 0)/10000) 
              for k in set(ideal_counts) | set(noisy_counts)}

# Print outcomes with their ideal probabilities, noisy probabilities, and difference
print("Outcome | Ideal Probability | Noisy Probability | Probability Difference")
for outcome in sorted(noise_prob.keys()):
    ideal_prob = ideal_counts.get(outcome, 0)/10000
    noisy_prob_val = noisy_counts.get(outcome, 0)/10000
    diff = noise_prob[outcome]
    print(f"{outcome:>7} | {ideal_prob:17.4f} | {noisy_prob_val:16.4f} | {diff:21.4f}")

plt.figure(figsize=(10, 5))
plt.bar(noise_prob.keys(), noise_prob.values())
plt.title('Noise Probability per Outcome')
plt.xlabel('Outcome')
plt.ylabel('Probability Difference')
plt.grid(True)
plt.show()

# Plot histograms
plt.figure(figsize=(12, 6))

# Ideal results
plt.subplot(1, 2, 1)
plot_histogram(ideal_counts, title='Ideal Simulation Results', ax=plt.gca())

# Noisy results
plt.subplot(1, 2, 2)
plot_histogram(noisy_counts, title='Noisy Simulation Results (FakeManilaV2)', ax=plt.gca())

plt.tight_layout()
plt.show()
