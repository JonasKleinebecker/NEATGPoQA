from numpy import pi
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import QasmSimulator, StatevectorSimulator
import random
import numpy as np

num_shots = 1024

def create_deutsch_josza_oracle_gate(num_input_qubits, oracle_case):
    """
    Creates a quantum gate implementing a Deutsch-Jozsa oracle.

    Args:
        oracle_type (int): Specifies the case of oracle to create. 
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

    if oracle_case == 0:
        pass  

    elif oracle_case == 1:
        qc.x(qr[total_qubits-1])  # Flip the output qubit to 1

    elif oracle_case in range(2, 8):
        if num_input_qubits == 1:
            if(oracle_case > 3):
                raise ValueError("Invalid oracle_type. For the 1 input qubit variant there are only 4 possible functions.")
            outputs_map = [[0, 1], [1, 0]]  # For 1 input qubit
            outputs = outputs_map[oracle_case - 2]
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
            outputs = outputs_map[oracle_case - 2]

        for i, output in enumerate(outputs):
            if output == 1:
                for j in range(num_input_qubits):
                    if i & (1 << j):  # Check if the j-th bit of 'i' is 1
                        qc.cx(qr[j], qr[total_qubits-1])

    else:
        raise ValueError("Invalid oracle_type. Must be an integer between 0 and 8.")

    return qc

def run_circuit(simulator_type, individual, num_qubits, oracle_case, initial_bit_state = "0"):
    """Converts a DEAP individual into a Qiskit circuit and runs it on a simulator."""
    if simulator_type == "qasm":
        qc = QuantumCircuit(num_qubits, num_qubits)
    elif simulator_type == "statevector":
        qc = QuantumCircuit(num_qubits)
    else:
        raise ValueError("Invalid simulator_type. Choose 'qasm' or 'statevector'.") 

    # Prepare the initial state
    for j, bit in enumerate(initial_bit_state):
        if bit == '1':
            qc.x(j)

    # Apply gates from the individual to the circuit
    for gate_info in individual:
        gate_name, *args = gate_info
        if gate_name == "oracle":
            oracle_qc = create_deutsch_josza_oracle_gate(num_qubits - 1, oracle_case)
            qc.compose(oracle_qc, qubits=range(num_qubits), inplace=True)
        else:
            getattr(qc, gate_name)(*args)  
    
    if simulator_type == "qasm":
        qc.measure(range(num_qubits), range(num_qubits))
        simulator = QasmSimulator()
        job = simulator.run(qc, shots=num_shots)
        counts = job.result().get_counts(qc) 
        for key in counts:
            counts[key] /= num_shots
        return counts
    elif simulator_type == "statevector":
        simulator = StatevectorSimulator()
        job = simulator.run(qc)
        statevector = job.result().get_statevector(qc) 
        return statevector

def generate_3_qubit_fourier_transform_statevectors():
    """Prints the statevectors of the 3-qubit Fourier transform for all possible input states."""
    num_qubits = 3
    statevectors = {}
    for i in range(2**num_qubits):
        # Convert the integer 'i' to its binary representation
        initial_state_binary = format(i, '0' + str(num_qubits) + 'b')
        qc = QuantumCircuit(num_qubits)

        # Initialize the qubits to the corresponding initial state
        for j, bit in enumerate(initial_state_binary):
            if bit == '1':
                qc.x(j)

        # Apply the 3-qubit QFT
        qc.h(2)
        qc.cp(pi/2, 1, 2)
        qc.h(1)
        qc.cp(pi/4, 0, 2)
        qc.cp(pi/2, 0, 1)
        qc.h(0)
        qc.swap(0, 2) 

        simulator = StatevectorSimulator()
        job = simulator.run(qc)
        statevector = job.result().get_statevector(qc) 
        statevectors[initial_state_binary] = statevector
    return statevectors