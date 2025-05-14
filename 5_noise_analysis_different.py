from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt

def calculate_tanimoto_with_error(smiles1, smiles2, error_rate=0.01, shots=10000, mitigation=True):
    """
    Calculates Tanimoto similarity with depolarizing noise and improved mitigation.

    Args:
        smiles1 (str): SMILES string for the first molecule.
        smiles2 (str): SMILES string for the second molecule.
        error_rate (float): Depolarizing error rate.
        shots (int): Number of shots for the simulation.
        mitigation (bool): Whether to apply mitigation.

    Returns:
        tuple: Base Tanimoto similarity and noisy similarity.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string(s) provided")

    fp_gen = GetMorganGenerator(radius=2)
    fp1 = fp_gen.GetSparseFingerprint(mol1)
    fp2 = fp_gen.GetSparseFingerprint(mol2)
    base_similarity = TanimotoSimilarity(fp1, fp2)

    noise_model = NoiseModel()
    error_1q = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz'])
    error_2q = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(qr, cr)

    theta = 2 * np.arccos(np.sqrt(base_similarity)) if base_similarity > 0 else np.pi
    qc.ry(theta, qr[0])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[0], qr[2])
    qc.measure(qr, cr)

    simulator = AerSimulator(noise_model=noise_model)
    noisy_circuit = transpile(qc, simulator)
    job = simulator.run(noisy_circuit, shots=shots)
    result = job.result().get_counts()

    # Improved Similarity Estimation
    expectation_z0 = 0
    for outcome, count in result.items():
        sign = 1 if outcome[0] == '0' else -1  # Parity of the first qubit
        expectation_z0 += sign * count / shots

    # Mitigation (scaling)
    if mitigation:
        # Crude mitigation: Scale the expectation value back towards the ideal.
        # The scale factor depends on the error rate.
        scale_factor = np.exp(error_rate)  # Adjust as needed; this is a simple guess
        mitigated_expectation = expectation_z0 * scale_factor
        noisy_similarity = (1 + mitigated_expectation) / 2  # Map back to [0, 1]
    else:
        noisy_similarity = (1 + expectation_z0) / 2 # Map to [0, 1]

    return base_similarity, noisy_similarity

def plot_similarity_comparison(smiles1, smiles2, error_rates, mitigation=True):
    """Plots the comparison of base and noisy Tanimoto similarities across error rates."""
    base_sims = []
    noisy_sims = []

    for rate in error_rates:
        base_sim, noisy_sim = calculate_tanimoto_with_error(smiles1, smiles2, error_rate=rate, mitigation=mitigation)
        base_sims.append(base_sim)
        noisy_sims.append(noisy_sim)


    # Print out the XY values for the plot
    print("Error Rate\tBase Similarity\tNoisy Similarity")
    for er, bs, ns in zip(error_rates, base_sims, noisy_sims):
       print(f"{er:.5f}\t\t{bs:.4f}\t\t{ns:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(error_rates, base_sims, label='Base Similarity', marker='o', linestyle='--')
    plt.plot(error_rates, noisy_sims, label='Noisy Similarity', marker='s')
    plt.xlabel('Error Rate')
    plt.ylabel('Tanimoto Similarity')
    plt.title(f'Tanimoto Similarity vs Error Rate (Mitigation: {mitigation})')
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()


# Example usage
smiles1 = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
smiles2 = "CCCC"  # Butane

try:
    error_rates = np.logspace(-3, -1, 10)
    plot_similarity_comparison(smiles1, smiles2, error_rates, mitigation=False)
    plot_similarity_comparison(smiles1, smiles2, error_rates, mitigation=True)

except ValueError as e:
    print(f"Error: {e}")


try:
    error_rates = np.logspace(-3, -1, 10)
    for rate in error_rates:
        calculate_tanimoto_with_error(smiles1, smiles2, error_rate=rate, mitigation=False)
        calculate_tanimoto_with_error(smiles1, smiles2, error_rate=rate, mitigation=True)

except ValueError as e:
    print(f"Error: {e}")

# Example usage
smiles1 = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
smiles2 = "CCCC"  # Butane

try:
    error_rates = np.logspace(-3, -1, 10)
    for rate in error_rates:
        base_sim, noisy_sim = calculate_tanimoto_with_error(smiles1, smiles2, error_rate=rate, mitigation=False)
        print(f"Error Rate: {rate}, Base Tanimoto similarity: {base_sim:.4f}, Tanimoto similarity with error model (no mitigation): {noisy_sim:.4f}")
        
        base_sim, noisy_sim = calculate_tanimoto_with_error(smiles1, smiles2, error_rate=rate, mitigation=True)
        print(f"Error Rate: {rate}, Base Tanimoto similarity: {base_sim:.4f}, Tanimoto similarity with error model (with mitigation): {noisy_sim:.4f}")

except ValueError as e:
    print(f"Error: {e}")

