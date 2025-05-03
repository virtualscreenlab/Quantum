from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
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

base_similarity = 0.5
circuit = create_similarity_circuit(base_similarity)

# Visualize the circuit in color
circuit.draw('mpl')
