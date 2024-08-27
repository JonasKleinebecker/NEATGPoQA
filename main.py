import random
import numpy as np
from deap import creator, base, tools
from quantum_simulation import run_circuit_on_simulator, run_circuit_on_statevector_simulator, deutsch_jozsa_oracle_balanced, deutsch_jozsa_oracle_balanced_inverse, deutsch_jozsa_oracle_const_0, deutsch_jozsa_oracle_const_1

gate_set = ["h", "x", "cx", "rx", "ry", "rz", "oracle"]

num_qubits = 3
min_individual_length = 3
max_individual_length = 15

# Create an individual class to hold a list of gates
creator.create("minErrorFitness", base.Fitness, weights=(-1.0,))  
creator.create("Individual", list, fitness=creator.minErrorFitness)

def create_individual():
    individual = []

    length = random.randint(min_individual_length, max_individual_length)  

    for _ in range(length):
        gate_name = random.choice(gate_set)
        
        if gate_name == "cx":
            qubit1 = random.randint(0, num_qubits - 1)
            qubit2 = random.randint(0, num_qubits - 1)
            while qubit2 == qubit1:  
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
    if len(ind1) < 2 or len(ind2) < 2: 
        return ind1, ind2

    crossover_point = random.randrange(1, min(len(ind1), len(ind2)))

    # Exchange segments after the crossover point
    ind1[crossover_point:], ind2[crossover_point:] = (
        ind2[crossover_point:],
        ind1[crossover_point:],
    )
    return ind1, ind2

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
            attempt_counter = 0
            max_attempts = 10 # Prevent infinite loop
            while individual[index_to_change][0] == "oracle" and attempt_counter < max_attempts:  # Oracles have no parameters to change
                index_to_change = random.randrange(len(individual))
                attempt_counter += 1
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

def calculate_fitness_spector_1998(individual, simulation_results, correct_states):
    hits = len(simulation_results)
    correctness = 0
    accuracy = 0
    for counts, correct_states_per_case in zip(simulation_results, correct_states):
        for state in correct_states_per_case:
            accuracy += counts.get(state, 0)
        accuracy /= 1024
        if (accuracy >= 0.52):
            hits -= 1
        else:
            correctness += 0.52 - accuracy
    if hits > 1:
        correctness / hits
    if hits == 0:
        fitness = len(individual) / 1000
    else:
        fitness = correctness + hits
    return fitness,

def calculate_fitness_correctness_error(individual, simulation_results, correct_states):
    correctness = 0
    for counts, correct_states_per_case in zip(simulation_results, correct_states):
        for state in correct_states_per_case:
            correctness += counts.get(state, 0)
    correctness /= 1024 * len(simulation_results)
    return 1 - correctness,

toolbox = base.Toolbox()
toolbox.register("calculate_fitness", calculate_fitness_spector_1998)

def evaluate_deutsch_josza_1_input_qubits(individual):
    counts_balanced = run_circuit_on_simulator(individual, 2, 2)
    counts_balanced_inverse = run_circuit_on_simulator(individual, 2, 3)
    counts_const_0 = run_circuit_on_simulator(individual, 2, 0)
    counts_const_1 = run_circuit_on_simulator(individual, 2 ,1)
    simulation_results = [counts_balanced, counts_balanced_inverse, counts_const_0, counts_const_1]
    correct_states = [["01", "11"], ["01", "11"], ["10","00"], ["10","00"]]
    return toolbox.calculate_fitness(individual, simulation_results, correct_states)

def evaluate_deutsch_josza_2_input_qubits(individual):
    counts_balanced_1 = run_circuit_on_simulator(individual, 3, 2)
    counts_balanced_2 = run_circuit_on_simulator(individual, 3, 3)
    counts_balanced_3 = run_circuit_on_simulator(individual, 3, 4)
    counts_balanced_4 = run_circuit_on_simulator(individual, 3, 5)
    counts_balanced_5 = run_circuit_on_simulator(individual, 3, 6)
    counts_balanced_6 = run_circuit_on_simulator(individual, 3, 7)
    counts_const_0 = run_circuit_on_simulator(individual, 3, 0)
    counts_const_1 = run_circuit_on_simulator(individual, 3, 1)
    simulation_results = [counts_balanced_1, counts_balanced_2, counts_balanced_3, counts_balanced_4, counts_balanced_5, counts_balanced_6, counts_const_0, counts_const_1]
    correct_states = [["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["100","000"], ["100","000"],]
    return toolbox.calculate_fitness(individual, simulation_results, correct_states)

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
toolbox.register("evaluate", evaluate_deutsch_josza_2_input_qubits)

pop_size = 1000
num_generations = 30
best_ind = None

pop = toolbox.population(n=pop_size)

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(num_generations):
    print(f"-- Generation {g} --")

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit 

    pop[:] = offspring

    fits = [ind.fitness.values[0] for ind in pop]
    print(f"  Min error: {min(fits)}")
    print(f"  Max error: {max(fits)}")
    print(f"  Avg error: {np.mean(fits)}")
    print(f"  Std error: {np.std(fits)}")
    
    curr_best_ind = tools.selBest(pop, 1)[0]
    if best_ind is None or curr_best_ind.fitness.values[0] < best_ind.fitness.values[0]:
        best_ind = curr_best_ind

    if min(fits) <= 0.007:
        break
if(num_generations > 0):
    print("Best individual:", best_ind)
    print("Best individual's error:", best_ind.fitness.values[0])
    print("best_ind spector_1998: ",toolbox.evaluate(best_ind))
    toolbox.register("calculate_fitness", calculate_fitness_correctness_error)
    print("best_ind correctness error: ",toolbox.evaluate(best_ind))
known_solution=[['h', 0],['h', 1], ['x', 2], ['h', 2], ['oracle'], ['h', 0],['h', 1]]
#known_solution=[['h', 0],['x', 1], ['h', 1], ['oracle'], ['h', 0]]
print("known solution spector_1998: ",toolbox.evaluate(known_solution))
toolbox.register("calculate_fitness", calculate_fitness_correctness_error)
print("known solution correctness error: ",toolbox.evaluate(known_solution))