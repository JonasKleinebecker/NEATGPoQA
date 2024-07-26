import random
import numpy as np
from deap import creator, base, tools

gate_set = ["H", "X", "Y", "Z", "CNOT", "RX", "RY", "RZ"]

num_qubits = 4  
min_individual_length = 1
max_individual_length = 9

# Create an individual class to hold a list of gates
creator.create("minErrorFitness", base.Fitness, weights=(-1.0,))  
creator.create("Individual", list, fitness=creator.minErrorFitness)

def create_individual():
    individual = []

    # Random circuit length (number of gates)
    length = random.randint(min_individual_length, max_individual_length)  

    for _ in range(length):
        gate_name = random.choice(gate_set)
        
        # Special handling two qubit gates
        if gate_name == "CNOT":
            qubit1 = random.randint(0, num_qubits - 1)
            qubit2 = random.randint(0, num_qubits - 1)
            while qubit2 == qubit1:  # Ensure qubits are different
                qubit2 = random.randint(0, num_qubits - 1)
            individual.append((gate_name, qubit1, qubit2))
        elif gate_name in ["RX", "RY", "RZ"]:  # Handle rotation gates
            qubit = random.randint(0, num_qubits - 1)
            angle = np.random.uniform(0, 2 * np.pi)  # Random angle between 0 and 2π
            individual.append((gate_name, qubit, angle))
        else:  # Single qubit gates
            qubit = random.randint(0, num_qubits - 1)
            individual.append((gate_name, qubit))

    return creator.Individual(individual)

toolbox = base.Toolbox()
toolbox.register("individual", create_individual)

# Create an individual
individual1 = toolbox.individual()
print(individual1)
