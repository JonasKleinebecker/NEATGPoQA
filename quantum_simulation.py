from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
from qiskit_aer import QasmSimulator
import numpy as np

def deutsch_jozsa_oracle_const_0(state):
    """Oracle function for the constant 0 function."""
    return 0

def deutsch_jozsa_oracle_const_1(state):
    """Oracle function for the constant 1 function."""
    return 1

def deutsch_jozsa_oracle_balanced(state):
    """Oracle function for the balanced function."""
    return sum((map(int, state))) % 2 

def deutsch_jozsa_oracle_balanced_inverse(state):
    """Oracle function for the balanced function."""
    return (sum((map(int, state))) + 1) % 2 

def unity_oracle(state):
    """Oracle function for the unity function."""
    return 1

def create_oracle_gate(num_qubits, oracle_function):
    """Creates a custom Oracle gate from a given function."""
    matrix_size = 2 ** num_qubits  
    unitary_matrix = np.eye(matrix_size)

    for i in range(matrix_size):
        state = format(i, f"0{num_qubits}b")  # Convert index to binary string
        if oracle_function(state):
            unitary_matrix[i, i] = -1  # Flip phase for marked state

    return UnitaryGate(unitary_matrix, label="Oracle") 

def run_circuit_on_simulator(individual, num_qubits, oracle_function):
    """Converts a DEAP individual into a Qiskit circuit and runs it on a simulator."""
    qc = QuantumCircuit(num_qubits, num_qubits) # number of qubits, number of classical bits

    # Apply gates from the individual to the circuit
    for gate_info in individual:
        gate_name, *args = gate_info
        if gate_name == "cx":
            qc.cx(*args)  
        elif gate_name in ["rx", "ry", "rz"]:
            getattr(qc, gate_name)(*args)  
        elif gate_name == "oracle":
            oracle_gate = create_oracle_gate(num_qubits, oracle_function)
            qc.append(oracle_gate, range(num_qubits))
        else:
            getattr(qc, gate_name)(*args)  
    qc.measure(range(num_qubits), range(num_qubits))
    simulator = QasmSimulator()
    job = simulator.run(qc, shots=1024)
    counts = job.result().get_counts(qc) 
    return counts

def evaluate_deutsch_josza(individual):
    counts_balanced = run_circuit_on_simulator(individual, 2, deutsch_jozsa_oracle_balanced)
    counts_balanced_inverse = run_circuit_on_simulator(individual, 2, deutsch_jozsa_oracle_balanced_inverse)
    counts_const_0 = run_circuit_on_simulator(individual, 2, deutsch_jozsa_oracle_const_0)
    counts_const_1 = run_circuit_on_simulator(individual, 2 ,deutsch_jozsa_oracle_const_1)

    error_balanced = (counts_balanced.get("00", 0) + counts_balanced.get("10", 0)) / 1024 # qiskit uses big-endian bit ordering. 10 therefore refers to the state |01⟩
    error_balanced_inverse = (counts_balanced_inverse.get("00", 0) + counts_balanced_inverse.get("10", 0)) / 1024
    error_const_0 = (counts_const_0.get("01", 0) + counts_const_0.get("11", 0)) / 1024
    error_const_1 = (counts_const_1.get("01", 0) + counts_const_1.get("11", 0)) / 1024

    error = (error_balanced + error_balanced_inverse + error_const_0 + error_const_1) / 4
    return error,