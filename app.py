import streamlit as st
import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, phase_damping_error
from qiskit.quantum_info import entropy, DensityMatrix
from qiskit.quantum_info import state_fidelity, Statevector

st.set_page_config(
    page_title="Quantum Decoherence Lab",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Quantum Decoherence Dynamics: GHZ State")



st.sidebar.header("Parameters")

n_qubits = st.sidebar.slider(
    "Number of Qubits (N)", 
    min_value=2, 
    max_value=5, 
    value=3,
)


noise_level = st.sidebar.slider(
    "Decoherence Level (Phase Damping)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.0,
    step=0.01, 
    help="0.0 = Pure State . 1.0 = Mixed State ."
)

st.sidebar.divider()


phase_angle = st.sidebar.slider(
    "Apply Phase Rotation (Z-Gate)",
    min_value=0.0,
    max_value=np.pi,
    value=0.0,
    step=0.1,
    format="%.1f rad",
    help="Rotates the state around the Z-axis."
)


@st.cache_data
def run_simulation(n, noise, phase):
    qc = QuantumCircuit(n)
   
    qc.h(0)
    
    if phase > 0:
        qc.p(phase, 0) 
        
    for i in range(n - 1):
        qc.cx(i, i + 1)
    
    noise_model = NoiseModel()
    if noise > 0:
	
        if noise >= 1.0:
            gate_noise = 1.0
        else:
            
            gate_noise = 1 - (1 - noise)**(1/n)
        error_1q = phase_damping_error(gate_noise)
        error_2q = error_1q.tensor(error_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'id', 'rz', 'sx', 'x', 'p'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    
    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    qc.save_density_matrix()
    result = sim.run(transpile(qc, sim)).result()
    
    return result.data()['density_matrix']

def plot_interactive_density_matrix(rho, n, component='Real'):
    
 
    if component == 'Real Part':
        data = np.real(rho)
        z_range = [-0.5, 1.0] 
    elif component == 'Imaginary Part':
        data = np.imag(rho)
        z_range = [-0.5, 0.5] 
    else: # Absolute Value
        data = np.abs(rho)
        z_range = [0.0, 1.0]
    
    rows, cols = data.shape
    x_vals = np.arange(cols)
    y_vals = np.arange(rows)
    
    fig = go.Figure(data=[go.Surface(
        z=data,
        x=x_vals,
        y=y_vals,
        colorscale='Viridis',
        opacity=0.9,
        contours = {"z": {"show": True, "start": 0, "end": 1, "size": 0.05, "color":"white"}}
    )])
    
    dim = 2**n
    tick_vals = [0, dim-1]
    tick_text = [f"|{'0'*n}⟩", f"|{'1'*n}⟩"] 
    
    fig.update_layout(
        title=f"Density Matrix Topography ({component})",
        scene=dict(
            xaxis=dict(title='Ket (Col)', tickvals=tick_vals, ticktext=tick_text),
            yaxis=dict(title='Bra (Row)', tickvals=tick_vals, ticktext=tick_text),
            zaxis=dict(title='Amplitude', range=z_range),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )
    return fig


rho = run_simulation(n_qubits, noise_level, phase_angle)

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("State Metrics")
    
    
    purity = np.trace(np.dot(rho, rho)).real
    st.metric("System Purity", f"{purity:.4f}", delta_color="normal")
    
 
    rho_obj = DensityMatrix(rho)
    vn_entropy = entropy(rho_obj)
    st.metric("Von Neumann Entropy", f"{vn_entropy:.4f}", help="Quantifies information lost to the environment.")

    st.divider()
    
    st.info(f"""
    **Current Status:**
    * **N:** {n_qubits} Qubits
    * **Noise:** {noise_level:.2f}
    """)
  
    qc_ideal = QuantumCircuit(n_qubits)
    qc_ideal.h(0)
    
    
    if phase_angle > 0:
        qc_ideal.p(phase_angle, 0)
        
    for i in range(n_qubits - 1):
        qc_ideal.cx(i, i+1)
        
    target_state = Statevector.from_instruction(qc_ideal)

    
    fidelity = state_fidelity(target_state, rho)

    st.metric("State Fidelity", f"{fidelity:.4f}")

    if fidelity > 0.5:
        st.success(f"✅ Multipartite Entanglement (F={fidelity:.2f} > 0.5)")
    else:
        st.error(f"❌ Classical Limit Reached (F={fidelity:.2f} <= 0.5)")
with col2:
   
    view_mode = st.radio(
        "Visualization Mode:",
        ('Real Part', 'Imaginary Part', 'Absolute Value'),
        horizontal=True
    )
    
    fig = plot_interactive_density_matrix(rho, n_qubits, component=view_mode)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Developed by Abdullah Said Töngel")