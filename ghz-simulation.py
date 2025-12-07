import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel, phase_damping_error
from qiskit.visualization import plot_state_city

# --- CONFIGURATION ---
NUM_QUBITS = 5  # N=5 entanglement 
OUTPUT_DPI = 300 

def create_ghz_circuit(n_qubits):
    
    qc = QuantumCircuit(n_qubits)
    qc.h(0)  # Superposition
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)  
    
   
    qc.save_density_matrix()
    return qc

def run_simulation(qc, noise_prob):
    
    
    noise_model = NoiseModel()
    if noise_prob > 0:
      
        error = phase_damping_error(noise_prob)
        noise_model.add_all_qubit_quantum_error(error, ['cx', 'h', 'id'])
    
    
    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    
   
    t_qc = transpile(qc, sim)
    result = sim.run(t_qc).result()
    
    return result.data()['density_matrix']

def visualize_results(density_matrix, label, noise_val):
    
    print(f"   Rendering {label} (Noise={noise_val})...")
    
    title_text = f"Density Matrix: {label} (Noise $\lambda={noise_val}$)"
    
    fig = plot_state_city(density_matrix, title=title_text, figsize=(12, 8))
    
    filename = f"state_city_{label.lower().replace(' ', '_')}.png"
    fig.savefig(filename, dpi=OUTPUT_DPI)
    print(f"   Saved: {filename}")

def main():
    print(f"--- STARTING QUANTUM DECOHERENCE SIMULATION (N={NUM_QUBITS}) ---")
    
    
    qc = create_ghz_circuit(NUM_QUBITS)
    print(f"1. Circuit built for {NUM_QUBITS}-Qubit GHZ State.")

    
    print("\n2. Simulating Isolated System (Pure State)...")
    rho_pure = run_simulation(qc, noise_prob=0.0)
    visualize_results(rho_pure, "Pure State", 0.0)

    
    print("\n3. Simulating Open System (Environmental Interaction)...")
    rho_mixed = run_simulation(qc, noise_prob=0.5)
    visualize_results(rho_mixed, "Decohered State", 0.5)

    print("\n--- SIMULATION COMPLETE. CHECK FOLDER FOR IMAGES. ---")

if __name__ == "__main__":
    main()