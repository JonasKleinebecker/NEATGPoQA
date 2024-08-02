import random
import numpy as np
from deap import creator, base, tools
from qiskit import QuantumCircuit
from qiskit_aer import Aer, QasmSimulator
from qiskit.circuit.library import UnitaryGate

gate_set = ["h", "x", "y", "z", "cx", "rx", "ry", "rz", "oracle"]

num_qubits = 2  
min_individual_length = 1
max_individual_length = 9

# Create an individual class to hold a list of gates
creator.create("minErrorFitness", base.Fitness, weights=(-1.0,))  
creator.create("Individual", list, fitness=creator.minErrorFitness)

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

def create_individual():
    individual = []

    # Random circuit length (number of gates)
    length = random.randint(min_individual_length, max_individual_length)  

    for _ in range(length):
        gate_name = random.choice(gate_set)
        
        # Special handling two qubit gates
        if gate_name == "cx":
            qubit1 = random.randint(0, num_qubits - 1)
            qubit2 = random.randint(0, num_qubits - 1)
            while qubit2 == qubit1:  # Ensure qubits are different
                qubit2 = random.randint(0, num_qubits - 1)
            individual.append((gate_name, qubit1, qubit2))
        elif gate_name in ["rx", "ry", "rz"]:  
            qubit = random.randint(0, num_qubits - 1)
            angle = np.random.uniform(0, 2 * np.pi) 
            individual.append((gate_name, angle, qubit))
        elif gate_name == "oracle":
            individual.append([gate_name])
        else:  
            qubit = random.randint(0, num_qubits - 1)
            individual.append((gate_name, qubit))

    return creator.Individual(individual)

def single_point_crossover(ind1, ind2):
    """Performs single-point crossover on two individuals."""
    if len(ind1) < 2 or len(ind2) < 2:  # Ensure enough gates for crossover
        return ind1, ind2

    crossover_point = random.randrange(1, min(len(ind1), len(ind2)))

    # Exchange segments after the crossover point
    ind1[crossover_point:], ind2[crossover_point:] = (
        ind2[crossover_point:],
        ind1[crossover_point:],
    )
    return ind1, ind2

# --- Mutation Function ---
def mutate_individual(individual, mut_pb, add_pb, del_pb, change_pb, param_pb):
    """Mutates an individual based on multiple probabilities."""

    if random.random() < mut_pb:
        mutation_type = np.random.choice(["add", "delete", "change", "params"], 
                                         p=[add_pb, del_pb, change_pb, param_pb])

        if mutation_type == "add":
            new_gate = create_individual()[0]  
            insert_point = random.randint(0, len(individual))
            individual.insert(insert_point, new_gate)

        elif mutation_type == "delete" and len(individual) > 1:  # Ensure at least one gate remains
            del individual[random.randrange(len(individual))]

        elif mutation_type == "change":
            index_to_change = random.randrange(len(individual))
            while individual[index_to_change][0] == "oracle":  # Oracles have no parameters to change
                index_to_change = random.randrange(len(individual))
            individual[index_to_change] = create_individual()[0]

        elif mutation_type == "params": 
            index_to_change = random.randrange(len(individual))
            gate_name, *args = individual[index_to_change]
            if gate_name == "cx":
                new_qubit1 = random.randint(0, num_qubits - 1)
                new_qubit2 = random.randint(0, num_qubits - 1)
                while new_qubit1 == new_qubit2: 
                    new_qubit2 = random.randint(0, num_qubits - 1)
                individual[index_to_change] = (gate_name, new_qubit1, new_qubit2)

            elif gate_name in ["rx", "ry", "rz"]:
                new_angle = np.random.uniform(0, 2 * np.pi)
                new_qubit = random.randint(0, num_qubits - 1)
                individual[index_to_change] = (gate_name, new_angle, new_qubit)

            else:
                new_qubit = random.randint(0, num_qubits - 1)
                individual[index_to_change] = (gate_name, new_qubit)

    return individual, # Comma is needed because the function should return a tuple due to DEAP requirements

def run_circuit_on_simulator(individual, oracle_function):
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
    counts_balanced = run_circuit_on_simulator(individual, deutsch_jozsa_oracle_balanced)
    counts_balanced_inverse = run_circuit_on_simulator(individual, deutsch_jozsa_oracle_balanced_inverse)
    counts_const_0 = run_circuit_on_simulator(individual, deutsch_jozsa_oracle_const_0)
    counts_const_1 = run_circuit_on_simulator(individual, deutsch_jozsa_oracle_const_1)

    # Calculate the error in the Deutsch-Josza algorithm
    error_balanced = (counts_balanced.get("00", 0) + counts_balanced.get("10", 0)) / 1024 # qiskit uses big-endian bit ordering. 10 therefore refers to the state |01⟩
    error_balanced_inverse = (counts_balanced_inverse.get("00", 0) + counts_balanced_inverse.get("10", 0)) / 1024
    error_const_0 = (counts_const_0.get("01", 0) + counts_const_0.get("11", 0)) / 1024
    error_const_1 = (counts_const_1.get("01", 0) + counts_const_1.get("11", 0)) / 1024

    error = (error_balanced + error_balanced_inverse + error_const_0 + error_const_1) / 4
    return error,

toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) 
toolbox.register("mate", single_point_crossover)
mut_pb = 0.2  # Overall mutation probability
add_pb = 0.25
del_pb = 0.25
change_pb = 0.25
param_pb = 0.25

toolbox.register("mutate", mutate_individual, mut_pb=mut_pb, add_pb=add_pb,
                 del_pb=del_pb, change_pb=change_pb, param_pb=param_pb)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", evaluate_deutsch_josza)

# Define Population Size and Number of Generations
pop_size = 100
num_generations = 1
best_ind = None

# Create Initial Population
pop = toolbox.population(n=pop_size)

# Evaluate Initial Fitness
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# tests
individual = [('y', 0), ('y', 0), ('rx', 3.4273755036537255, 1), ('h', 0), ['oracle'], ('h', 0), ('cx', 0, 1)]
counts = run_circuit_on_simulator(individual, deutsch_jozsa_oracle_const_1)
counts_00 = counts.get("00", 0) / 1024
counts_01 = counts.get("01", 0) / 1024
counts_10 = counts.get("10", 0) / 1024
counts_11 = counts.get("11", 0) / 1024
print(counts_00, counts_01, counts_10, counts_11)

# check correct solution for the problem
correct_solution = [["x", 1],["h", 0], ["h", 1], ["oracle"], ["h", 0]]
counts = run_circuit_on_simulator(correct_solution, deutsch_jozsa_oracle_const_1)
counts_00 = counts.get("00", 0) / 1024
counts_01 = counts.get("01", 0) / 1024
counts_10 = counts.get("10", 0) / 1024
counts_11 = counts.get("11", 0) / 1024
print(counts_00, counts_01, counts_10, counts_11)

print(evaluate_deutsch_josza(correct_solution))

# Run Genetic Algorithm
for g in range(num_generations):
    print(f"-- Generation {g} --")

    # Select Parents
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # Apply Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply Mutation
    for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate Fitness of New Individuals
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit 

    # Replace Population
    pop[:] = offspring

    # Gather and Print Statistics (Optional)
    fits = [ind.fitness.values[0] for ind in pop]
    print(f"  Min error: {min(fits)}")
    print(f"  Max error: {max(fits)}")
    print(f"  Avg error: {np.mean(fits)}")
    print(f"  Std error: {np.std(fits)}")
    
    curr_best_ind = tools.selBest(pop, 1)[0]
    if best_ind is None or curr_best_ind.fitness.values[0] < best_ind.fitness.values[0]:
        best_ind = curr_best_ind

#Print Best Individual
print("Best individual:", best_ind)
print("Best individual's error:", best_ind.fitness.values[0])

