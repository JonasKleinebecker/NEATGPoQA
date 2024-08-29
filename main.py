import random
import time
import numpy as np
import deap.tools.emo as emo
from deap import creator, base, tools, algorithms
from qiskit_aer.backends.compatibility import Statevector
from quantum_simulation import run_circuit, generate_3_qubit_fourier_transform_statevectors

gate_set = ["h", "x", "cx", "rx", "ry", "rz", "swap", "cp"]

num_qubits = 3
min_individual_length = 3
max_individual_length = 30
use_multi_objective = False
simulator_type = "statevector"
statevector_holder = generate_3_qubit_fourier_transform_statevectors()

# Create an individual class to hold a list of gates
if(use_multi_objective):
    creator.create("minErrorFitness", base.Fitness, weights=(-1.0, -1.0))  # Minimize error, minimize number of gates
else:
    creator.create("minErrorFitness", base.Fitness, weights=(-1.0,))  
creator.create("Individual", list, fitness=creator.minErrorFitness)

def create_gate():
    gate_name = random.choice(gate_set)
    control_qubit = None  # Needs to be remembered to avoid duplicate control and target qubits

    gate_args = []
    gate_args.append(gate_name)
    if gate_name in ["rx", "ry", "rz", "cp"]:
        angle = np.random.uniform(0, 2 * np.pi)
        gate_args.append(angle)
    if gate_name in ["cx", "swap", "cp"]:
        control_qubit = random.randint(0, num_qubits - 1)
        gate_args.append(control_qubit)
    if gate_name in ["h", "x", "y", "z", "cx", "cp", "swap", "rx", "ry", "rz"]:
        target_qubit = random.randint(0, num_qubits - 1)
        while control_qubit == target_qubit:
            target_qubit = random.randint(0, num_qubits - 1)
        gate_args.append(target_qubit)
    return gate_args

def create_individual():
    individual = []

    length = random.randint(min_individual_length, max_individual_length)  

    for _ in range(length):
        individual.append(create_gate())
    return creator.Individual(individual)

# ------- Genetic operators -------
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
            new_gate = create_gate()  
            insert_point = random.randint(0, len(individual))
            individual.insert(insert_point, new_gate)

        elif mutation_type == "delete" and len(individual) > 1:  # Ensure at least one gate remains
            del individual[random.randrange(len(individual))]

        elif mutation_type == "change":
            index_to_change = random.randrange(len(individual))
            individual[index_to_change] = create_gate()

        elif mutation_type == "params": 
            index_to_change = random.randrange(len(individual))
            gate_name, *args = individual[index_to_change]
            if gate_name in ["cx", "swap"]:
                new_qubit1 = random.randint(0, num_qubits - 1)
                new_qubit2 = random.randint(0, num_qubits - 1)
                while new_qubit1 == new_qubit2: 
                    new_qubit2 = random.randint(0, num_qubits - 1)
                individual[index_to_change] = ([gate_name, new_qubit1, new_qubit2])

            elif gate_name in ["cp"]:
                new_angle = np.random.uniform(0, 2 * np.pi)
                new_qubit1 = random.randint(0, num_qubits - 1)
                new_qubit2 = random.randint(0, num_qubits - 1)
                while new_qubit1 == new_qubit2: 
                    new_qubit2 = random.randint(0, num_qubits - 1)
                individual[index_to_change] = ([gate_name, new_angle, new_qubit1, new_qubit2])

            elif gate_name in ["rx", "ry", "rz"]:
                new_angle = np.random.uniform(0, 2 * np.pi)
                new_qubit = random.randint(0, num_qubits - 1)
                individual[index_to_change] = ([gate_name, new_angle, new_qubit])

            else:
                new_qubit = random.randint(0, num_qubits - 1)
                individual[index_to_change] = ([gate_name, new_qubit])

    return individual, # Comma is needed because the function should return a tuple due to DEAP requirements

#------- Helper functions -------
def normalize_value_0_to_1(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def convert_statevectors_to_probabilities_dict(statevectors):
    results_probabilities = []
    for statevector in statevectors:    
        statevector_array = np.asarray(statevector) # Working on the statevector directly is deprecated
        probabilities_dict = {}
        for i, amplitude in enumerate(statevector_array):
            probability = abs(amplitude) ** 2
            bitstring = format(i, f'0{num_qubits}b')  # Convert index to bitstring
            probabilities_dict[bitstring] = probability
        results_probabilities.append(probabilities_dict)
    return results_probabilities

#------- Single-objective fitness functions -------
def calculate_spector_1998_fitness(individual, simulation_results, correct_states):
    hits = len(simulation_results)
    correctness = 0
    if(isinstance(simulation_results[0],Statevector)):
        results_probabilities = convert_statevectors_to_probabilities_dict(simulation_results)
        simulation_results = results_probabilities
    for simulation_result, correct_states_per_case in zip(simulation_results, correct_states):
        accuracy = 0
        for state in correct_states_per_case:
            accuracy += simulation_result.get(state, 0)
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

def calculate_custom_spector_1998_fitness(individual, simulation_results, correct_states):
    hits = len(simulation_results)
    error = 0
    if(isinstance(simulation_results[0],Statevector)):
        results_probabilities = convert_statevectors_to_probabilities_dict(simulation_results)
        simulation_results = results_probabilities
    for simulation_result, correct_states_per_case in zip(simulation_results, correct_states):
        accuracy = 0
        for state in correct_states_per_case:
            accuracy += simulation_result.get(state, 0)
        error += 1 - accuracy
        if (accuracy >= 0.52):
            hits -= 1
    error /= len(simulation_results)
    fitness = error + hits
    if fitness < 0.00001: # account for floating point errors
        fitness = len(individual) / 1000
    return fitness,

def calculate_error_fitness(individual, simulation_results, correct_states):
    correctness = 0
    if(isinstance(simulation_results[0],Statevector)):
        results_probabilities = convert_statevectors_to_probabilities_dict(simulation_results)
        simulation_results = results_probabilities
    for simulation_result, correct_states_per_case in zip(simulation_results, correct_states):
        for state in correct_states_per_case:
            correctness += simulation_result.get(state, 0)
    correctness /= len(simulation_results) 
    return 1 - correctness,

def calculate_phase_aware_error_fitness(individual, simulation_statevectors, correct_statevectors): 
    if not isinstance(simulation_statevectors[0], Statevector):
        raise ValueError("The called 'calculate_phase_aware_error_fitness' function is only applicable for statevectors.")
    correctness = 0
    for simulation_statevector, correct_statevector in zip(simulation_statevectors, correct_statevectors.values()):
        correctness += np.abs(np.dot(simulation_statevector, correct_statevector.conjugate())) 
    correctness /= len(simulation_statevectors)
    return 1 - correctness,

#------- Multi-objective fitness functions -------
def calculate_error_and_gate_count(individual, simulation_results, correct_states):
    error = calculate_error_fitness(individual, simulation_results, correct_states)
    return error[0], normalize_value_0_to_1(len(individual), 1, max_individual_length) # Even though the min_individual_length is 3, mutation can reduce the length to a min of 1

toolbox = base.Toolbox()
toolbox.register("calculate_fitness", calculate_phase_aware_error_fitness)

def evaluate_deutsch_josza_1_input_qubits_qasm(individual):
    counts_balanced = run_circuit(simulator_type, individual, 2, 2)
    counts_balanced_inverse = run_circuit(simulator_type, individual, 2, 3)
    counts_const_0 = run_circuit(simulator_type, individual, 2, 0)
    counts_const_1 = run_circuit(simulator_type, individual, 2 ,1)
    simulation_results = [counts_balanced, counts_balanced_inverse, counts_const_0, counts_const_1]
    correct_states = [["01", "11"], ["01", "11"], ["10","00"], ["10","00"]]
    return toolbox.calculate_fitness(individual, simulation_results, correct_states)



def evaluate_deutsch_josza_2_input_qubits(individual):
    results_const_0 = run_circuit(simulator_type, individual, 3, 0)
    results_const_1 = run_circuit(simulator_type, individual, 3, 1)
    results_balanced_1 = run_circuit(simulator_type, individual, 3, 2)
    results_balanced_2 = run_circuit(simulator_type, individual, 3, 3)
    results_balanced_3 = run_circuit(simulator_type, individual, 3, 4)
    results_balanced_4 = run_circuit(simulator_type, individual, 3, 5)
    results_balanced_5 = run_circuit(simulator_type, individual, 3, 6)
    results_balanced_6 = run_circuit(simulator_type, individual, 3, 7)
    simulation_results = [results_balanced_1, results_balanced_2, results_balanced_3, results_balanced_4, results_balanced_5, results_balanced_6, results_const_0, results_const_1]
    correct_states = [["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["001","010","101","110","011", "111"], ["100","000"], ["100","000"],]
    return toolbox.calculate_fitness(individual, simulation_results, correct_states)

def evaluate_quantum_fourier_3_qubits(individual):
    simulation_results = []
    for i in range (8):
        initial_state_binary = format(i, '0' + str(3) + 'b')
        simulation_results.append(run_circuit(simulator_type, individual, 3, None, initial_state_binary))
    return toolbox.calculate_fitness(individual, simulation_results, statevector_holder)

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
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_quantum_fourier_3_qubits)

pop_size = 1000
num_generations = 100
num_elites = 25

def run_gp():
    best_ind = None
    pop = toolbox.population(n=pop_size)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(num_generations):
        print(f"-- Generation {g + 1} --")
        start_time_gp = time.time()

        elites = tools.selBest(pop, num_elites)

        offspring = toolbox.select(pop, len(pop) - num_elites)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        start_time_sim = time.time()
        fitnesses = map(toolbox.evaluate, invalid_ind)
        print(f"  Time taken for simulation: {time.time() - start_time_sim:.2f}s")
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit 

        pop[:] = offspring + elites

        # print fitness statistics
        num_objectives = len(pop[0].fitness.values)
        for i in range(num_objectives):
            fits = [ind.fitness.values[i] for ind in pop]
            print(f"  Min error of objective {i + 1}: {min(fits)}")
            print(f"  Max error of objective {i + 1}: {max(fits)}")
            print(f"  Avg error of objective {i + 1}: {np.mean(fits)}")
    
        # keep track of the best individual in single objective case
        if num_objectives == 1:
            curr_best_ind = tools.selBest(pop, 1)[0]
            if best_ind is None or curr_best_ind.fitness.values[0] < best_ind.fitness.values[0]:
                best_ind = curr_best_ind

        print(f"  Time taken for generation: {time.time() - start_time_gp:.2f}s")

    if num_objectives > 1:
        selected_pop = tools.selNSGA2(pop, len(pop))
        # Perform non-dominated sorting to extract the Pareto front
        pareto_fronts = tools.sortNondominated(selected_pop, len(selected_pop))
        best_ind = pareto_fronts[0] 
    if(num_generations > 0):
        if(num_objectives == 1):
            print(f"Best individual: {best_ind}")
            print(f"Best individual error: {best_ind.fitness.values[0]}")
        else:
            for ind_idx, ind in enumerate(best_ind):
                for obj_idx ,obj_value in enumerate(ind.fitness.values):
                    print(f"Best individual {ind_idx + 1}, {ind}")
                    print(f"Best individual {ind_idx + 1}, objective {obj_idx + 1}: {obj_value}")
                print(f"Individual {ind_idx + 1} gate count: {len(ind)}")

quantum_fourier_3_individual = [["h", 2], ["cp", np.pi/2, 1, 2], ["h", 1], ["cp", np.pi/4, 0, 2], ["cp",np.pi/2, 0, 1], ["h", 0], ["swap", 0, 2]]
deutsch_josza_3_individual = [('h', 0), ('h', 1), ('x', 2), ('h', 2), ['oracle'], ('h', 0), ('h', 1)]
print(f"Known solution error: {toolbox.evaluate(quantum_fourier_3_individual)}")
run_gp()
