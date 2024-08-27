from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import QasmSimulator, StatevectorSimulator
import random
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

def create_oracle_gate(num_qubits, oracle_type):
    """Creates a custom Oracle gate from a given function."""
    matrix_size = 2 ** num_qubits  
    unitary_matrix = np.eye(matrix_size)
    if(oracle_type == 1):
        unitary_matrix[[0,1,2,3]] = unitary_matrix[[1,0,3,2]]
    elif(oracle_type == 2):
        unitary_matrix[[0,1,2,3]] = unitary_matrix[[0,3,2,1]]
    elif(oracle_type == 3):
        unitary_matrix[[0,1,2,3]] = unitary_matrix[[2,0,3,1]]

    return UnitaryGate(unitary_matrix, label="Oracle") 

def create_deutsch_josza_oracle_gate(num_input_qubits, oracle_type):
    """
    Creates a quantum gate implementing a Deutsch-Jozsa oracle.

    Args:
        oracle_type (int): Specifies the type of oracle to create. 
                           0 = constant 0 function
                           1 = constant 1 function
                           2-8 = different balanced functions
        num_input_qubits (int): The number of input qubits (1 or 2). Defaults to 2.

    Returns:
        Gate: A quantum gate representing the chosen oracle.
    """

    if num_input_qubits not in [1, 2]:
        raise ValueError("Invalid num_input_qubits. Choose 1 or 2.")

    total_qubits = num_input_qubits + 1  # input qubits + 1 output qubit
    qr = QuantumRegister(total_qubits, 'q')
    qc = QuantumCircuit(qr)

    if oracle_type == 0:
        pass  

    elif oracle_type == 1:
        qc.x(qr[total_qubits-1])  # Flip the output qubit to 1

    elif oracle_type in range(2, 8):
        if num_input_qubits == 1:
            if(oracle_type > 3):
                raise ValueError("Invalid oracle_type. For the 1 input qubit variant there are only 4 possible functions.")
            outputs_map = [[0, 1], [1, 0]]  # For 1 input qubit
            outputs = outputs_map[oracle_type - 2]
            if outputs[0] == 1:
                qc.cx(qr[0], qr[total_qubits - 1]) 
        else:  
            outputs_map = [
                [0, 0, 1, 1], 
                [0, 1, 0, 1], 
                [0, 1, 1, 0], 
                [1, 0, 0, 1], 
                [1, 0, 1, 0], 
                [1, 1, 0, 0]]
            outputs = outputs_map[oracle_type - 2]

        for i, output in enumerate(outputs):
            if output == 1:
                for j in range(num_input_qubits):
                    if i & (1 << j):  # Check if the j-th bit of 'i' is 1
                        qc.cx(qr[j], qr[total_qubits-1])

    else:
        raise ValueError("Invalid oracle_type. Must be an integer between 0 and 8.")

    return qc

def run_circuit_on_simulator(individual, num_qubits, oracle_type):
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
            oracle_qc = create_deutsch_josza_oracle_gate(num_qubits - 1, oracle_type)
            qc.compose(oracle_qc, qubits=range(num_qubits), inplace=True)
        else:
            getattr(qc, gate_name)(*args)  
    qc.measure(range(num_qubits), range(num_qubits))
    simulator = QasmSimulator()
    job = simulator.run(qc, shots=1024)
    counts = job.result().get_counts(qc) 
    return counts

def run_circuit_on_statevector_simulator(individual, num_qubits, oracle_function):
    """Converts a DEAP individual into a Qiskit circuit and runs it on a statevector simulator."""
    qc = QuantumCircuit(num_qubits)  # Only qubits are needed, no classical bits

    # Apply gates from the individual to the circuit
    for gate_info in individual:
        gate_name, *args = gate_info
        if gate_name == "cx":
            qc.cx(*args) 
        elif gate_name in ["rx", "ry", "rz"]:
            getattr(qc, gate_name)(*args) 
        elif gate_name == "oracle":
            oracle_qc = create_oracle_gate(num_qubits, oracle_function)
            qc.compose(oracle_qc, qubits=range(num_qubits), inplace=True)
        else:
            getattr(qc, gate_name)(*args) 

    # No measurement is performed 

    simulator = StatevectorSimulator()
    job = simulator.run(qc)
    statevector = job.result().get_statevector(qc) 
    return statevector
