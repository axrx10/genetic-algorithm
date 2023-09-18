'''

This code implements a Genetic Algorithm (GA) framework that tests different parameter combinations across multiple generations and tracks the performance in terms of best and average fitness. The individuals in the population represent solutions to a given optimization problem.

The Individual class defines an individual in the population with genes and a fitness value. The create_initial_population function initializes the population with random gene values and calculates the fitness of each individual.

The polynomial_quartic function and rosenbrock_like_function are fitness functions that evaluate the quality of solutions. Lower fitness values represent better solutions.

tournament_selection and roulette_wheel_selection functions perform the selection operation for the GA. In tournament selection, two individuals are chosen at random and the fitter one is selected. In roulette wheel selection, individuals are selected based on their relative fitnesses.

Three types of crossover functions are defined: uniform_crossover, single_point_crossover, and two_point_crossover. These functions mix the genes of two parent individuals to generate offspring.

Two types of mutation functions are defined: uniform_mutation and random_resetting_mutation. The mutation operation introduces variability in the population by slightly altering the genes of the individuals.

The script then initializes some parameters, like the population size (P), number of genes per individual (N), and number of generations (GENERATIONS). It also prepares a list of selection, crossover, mutation, and fitness functions to be tested. It includes two types of selection methods, three types of crossover methods, two types of mutation methods, and two fitness functions.

Then, for a predefined number of sets (SET_OF_GENERATIONS), the GA runs for a number of generations, selecting one method from each list at random. It also randomly selects mutation rate (MUTRATE) and mutation step size (MUTSTEP). All these details are printed and written into a file.

Each generation involves performing selection, crossover, and mutation operations on the population. The fittest individual from each generation is preserved (elitism) and replaces the least fit individual in the next generation. At the end of each generation, the best and average fitnesses of the population are computed and stored. The process continues for a given number of generations.

Finally, the code visualizes the best and average fitness across each set of generations, and overall across all sets, using matplotlib. The generation and set that achieved the best overall fitness are also recorded. The best fitness achieved over all generations and all sets is printed out at the end.

'''

import random
import copy
import matplotlib.pyplot as plt
import math
import itertools

class Individual:
    def __init__(self, N, MIN, MAX):
        self.gene = [random.uniform(MIN, MAX) for _ in range(N)]
        self.fitness = 0

def create_initial_population(P, N, MIN, MAX, fitness_function):
    population = []
    for _ in range(P):
        newind = Individual(N, MIN, MAX)
        newind.fitness = fitness_function(newind, N)
        population.append(newind)
    return population

def rosenbrock_like_function(ind, N):
    return (ind.gene[0] - 1)**2 + sum((i+1) * (2 * ind.gene[i]**2 - ind.gene[i-1])**2 for i in range(1, N))

def polynomial_quartic(ind, N):
    return 0.5 * sum((ind.gene[i])**4 - (16*ind.gene[i])**2 + (5*ind.gene[i]) for i in range(N))


def tournament_selection(population, P):
    offspring = []
    for i in range(P):
        parent1 = random.randint(0, P-1)
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint(0, P-1)
        off2 = copy.deepcopy(population[parent2])
        if off1.fitness < off2.fitness:
            offspring.append(off1)
        else:
            offspring.append(off2)
    return offspring

def roulette_wheel_selection(population, P):
    total_fitness = sum(1.0 / individual.fitness for individual in population)
    selection_probs = [(1.0 / individual.fitness) / total_fitness for individual in population]

    offspring = []

    for _ in range(P):
        selected = random.choices(
            population=population,
            weights=selection_probs,
            k=1
        )[0]
        offspring.append(copy.deepcopy(selected))

    return offspring

def uniform_crossover(offspring, P, N):
    for i in range(0, P, 2):
        child1 = copy.deepcopy(offspring[i])
        child2 = copy.deepcopy(offspring[i+1])
        for j in range(N):
            if random.random() < 0.5:
                child1.gene[j], child2.gene[j] = child2.gene[j], child1.gene[j]
        offspring[i] = copy.deepcopy(child1)
        offspring[i+1] = copy.deepcopy(child2)
    return offspring

def single_point_crossover(offspring, P, N):
    for i in range(0, P, 2):
        child1 = copy.deepcopy(offspring[i])
        child2 = copy.deepcopy(offspring[i+1])
        crossover_point = random.randint(0, N-1)
        child1.gene[crossover_point:], child2.gene[crossover_point:] = child2.gene[crossover_point:], child1.gene[crossover_point:]
        offspring[i] = child1
        offspring[i+1] = child2
    return offspring


def two_point_crossover(offspring, P, N):
    for i in range(0, P, 2):
        child1 = copy.deepcopy(offspring[i])
        child2 = copy.deepcopy(offspring[i+1])
        crossover_point1 = random.randint(0, N-1)
        crossover_point2 = random.randint(0, N-1)
        if crossover_point2 < crossover_point1:
            crossover_point1, crossover_point2 = crossover_point2, crossover_point1
        child1.gene[crossover_point1:crossover_point2], child2.gene[crossover_point1:crossover_point2] = child2.gene[crossover_point1:crossover_point2], child1.gene[crossover_point1:crossover_point2]
        offspring[i] = child1
        offspring[i+1] = child2
    return offspring

def uniform_mutation(offspring, P, N, MUTRATE, MIN, MAX):
    ELITISM_MUTRATE = 0.01  # mutation rate for the elite individual
    for i in range(P):
        newind = Individual(N, MIN, MAX)
        for j in range(N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if i == 0:  # the best individual (elitism)
                if mutprob < ELITISM_MUTRATE:  # use a smaller mutation rate
                    gene += random.uniform(-MUTSTEP, MUTSTEP)
                    gene = max(min(gene, MAX), MIN)  # Ensure gene is within bounds
            else:  # the rest individuals
                if mutprob < MUTRATE:
                    gene += random.uniform(-MUTSTEP, MUTSTEP)
                    gene = max(min(gene, MAX), MIN)  # Ensure gene is within bounds
            newind.gene[j] = gene
        newind.fitness = fitness_function(newind, N)
        offspring[i] = copy.deepcopy(newind)
    return offspring


def random_resetting_mutation(offspring, P, N, MUTRATE, MIN, MAX):
    for i in range(P):
        for j in range(N):
            mutprob = random.random()
            if mutprob < MUTRATE:
                offspring[i].gene[j] = random.uniform(MIN, MAX)  # Uniformly random new value
        offspring[i].fitness = fitness_function(offspring[i], N)  # Recalculate fitness
    return offspring


def total_fitness(population):
    total = 0
    for individual in population:
        total += individual.fitness
    return total

N = 20
P = 5000
GENERATIONS = 100
SET_OF_GENERATIONS = 432
set_generations = []
set_best_fitness = []
set_average_fitness = []
best_of_bests = float('inf')
# Create a dictionary to store the best results of each set
best_results_per_set = {}

selection_functions = [tournament_selection, roulette_wheel_selection]
crossover_functions = [uniform_crossover, single_point_crossover, two_point_crossover]
mutation_functions = [uniform_mutation, random_resetting_mutation]
fitness_functions = [polynomial_quartic, rosenbrock_like_function]
mutation_rates = [0.5, 0.25, 0.12, 0.06, 0.03, 0.015]
mutation_steps = [5, 2.5, 1.25, 0.625, 0.3125, 0.15625]
# Define the fitness function to bounds mapping
fitness_bounds = {
    polynomial_quartic: (-5, 5),
    rosenbrock_like_function: (-10, 10),
}


# Generate all combinations
combinations = list(itertools.product(
    selection_functions, 
    crossover_functions, 
    mutation_functions, 
    fitness_functions, 
    mutation_rates, 
    mutation_steps
))


with open('Big run MAIN.txt', 'a') as f:
    for set_gen in range(SET_OF_GENERATIONS):
        generations = []
        best_fitness = []
        average_fitness = []
        best_fitness_offspring = float('inf')

        # Get a combination
        selection_function, crossover_function, mutation_function, fitness_function, MUTRATE, MUTSTEP = combinations.pop(random.randint(0, len(combinations) - 1))
        MIN, MAX = fitness_bounds[fitness_function]

        # Print the selected options at the start of the run
        print(f"Run {set_gen + 1} Selections:")
        print(f"Selection Function: {selection_function.__name__}")
        print(f"Crossover Function: {crossover_function.__name__}")
        print(f"Mutation Function: {mutation_function.__name__}")
        print(f"Fitness Function: {fitness_function.__name__}")
        print(f"Mutation Rate: {MUTRATE}")
        print(f"Mutation Step: {MUTSTEP}")    
        print(f"Bounds: {MIN} to {MAX}")                  
        
        # Write these values to the file
        f.write("\n")
        f.write(f"Run {set_gen + 1} Selections:\n")
        f.write(f"Selection Function: {selection_function.__name__}\n")
        f.write(f"Crossover Function: {crossover_function.__name__}\n")
        f.write(f"Mutation Function: {mutation_function.__name__}\n")
        f.write(f"Fitness Function: {fitness_function.__name__}\n")
        f.write(f"Mutation Rate: {MUTRATE}\n")
        f.write(f"Mutation Step: {MUTSTEP}\n")
        f.write(f"Bounds: {MIN} to {MAX}\n")

        # Store the combination and best fitness for this set
        best_results_per_set[set_gen] = {
        "selection_function": selection_function.__name__,
        "crossover_function": crossover_function.__name__,
        "mutation_function": mutation_function.__name__,
        "fitness_function": fitness_function.__name__,
        "mutation_rate": MUTRATE,
        "mutation_step": MUTSTEP,
        "best_fitness": float('inf'),  # This will be updated later
        "best_fitness_generation": -1,  # This will be updated later
        }
        population = create_initial_population(P, N, MIN, MAX, fitness_function)

        for gen in range(GENERATIONS):
            # Elitism: retain the best individual from the current generation
            best_individual = min(population, key=lambda ind: ind.fitness)

            offspring = selection_function(population, P)

            offspring = crossover_function(offspring, P, N)

            # Replace the worst individual in the new generation with the best individual from the previous generation
            worst_individual_index = max(range(P), key=lambda index: offspring[index].fitness)
            offspring[worst_individual_index] = copy.deepcopy(best_individual)
            offspring = uniform_mutation(offspring, P, N, MUTRATE, MIN, MAX)
            
            total_fitness_offspring = total_fitness(offspring)
            new_best_fitness_offspring = min(ind.fitness for ind in offspring)
            avg_fitness_offspring = total_fitness_offspring / P

            # Update the best fitness and the generation where it was achieved
            if new_best_fitness_offspring < best_results_per_set[set_gen]["best_fitness"]:
                best_results_per_set[set_gen]["best_fitness"] = new_best_fitness_offspring
                best_fitness_offspring = new_best_fitness_offspring
                best_results_per_set[set_gen]["best_fitness_generation"] = gen


            generations.append(gen)
            best_fitness.append(best_fitness_offspring)
            average_fitness.append(avg_fitness_offspring)

            # Update the best fitness and the generation and set where it was achieved
            if new_best_fitness_offspring < best_of_bests:
                best_of_bests = new_best_fitness_offspring
                best_generation = gen
                best_set = set_gen

            print(f"Generation: {gen+1} | Best Fitness: {best_fitness_offspring} | Average Fitness: {avg_fitness_offspring} | Mutation Rate: {MUTRATE} | Mutation Step Size: {MUTSTEP}")
            f.write(f"Generation: {gen+1} | Best Fitness: {best_fitness_offspring} | Average Fitness: {avg_fitness_offspring} | Mutation Rate: {MUTRATE} | Mutation Step Size: {MUTSTEP} \n")
            population = copy.deepcopy(offspring)

        # Adding average of averages and best of bests
        avg_of_avgs = sum(average_fitness) / len(average_fitness)
        best_of_gen_bests = min(best_fitness)
        print(f"Set: {set_gen+1} | Average Fitness: {avg_of_avgs} | Best Fitness: {best_of_gen_bests}\n")
        f.write(f"Set: {set_gen+1} | Average Fitness: {avg_of_avgs} | Best Fitness: {best_of_gen_bests}\n\n")

        # add results to the set results lists
        set_generations.append(generations)
        set_best_fitness.append(best_fitness)
        set_average_fitness.append(average_fitness)

        # Print the best results for each set
    for set_gen, results in best_results_per_set.items():
        print(f"Set {set_gen+1} | Best fitness: {results['best_fitness']} (achieved in generation {results['best_fitness_generation']+1})")
        f.write(f"Set {set_gen+1} | Best fitness: {results['best_fitness']} (achieved in generation {results['best_fitness_generation']+1})\n\n")
        print(f"  with parameters: selection function={results['selection_function']}, crossover function={results['crossover_function']}, mutation function={results['mutation_function']}, fitness function={results['fitness_function']}, mutation rate={results['mutation_rate']}, mutation step={results['mutation_step']}")
        f.write(f'''with parameters: 
                selection function={results['selection_function']}
                crossover function={results['crossover_function']}
                mutation function={results['mutation_function']}
                fitness function={results['fitness_function']}
                mutation rate={results['mutation_rate']}
                mutation step={results['mutation_step']}\n\n''')

n_sets_per_figure = 9 # The number of sets displayed per figure
n_figures = math.ceil(SET_OF_GENERATIONS / n_sets_per_figure) # The number of figures

for figure_i in range(n_figures):
    fig, axs = plt.subplots(3, 3, figsize=(10, 15))
    fig.suptitle(f'Rosenbrock Function (figure {figure_i + 1})')
    for j in range(n_sets_per_figure):
        set_i = figure_i * n_sets_per_figure + j # The actual set index
        if set_i >= SET_OF_GENERATIONS: # If we've exhausted all the sets
            break

        ax_row = j // 3
        ax_col = j % 3

        axs[ax_row, ax_col].plot(set_generations[set_i], set_best_fitness[set_i], label='Best Fitness')
        axs[ax_row, ax_col].set_title(f'Set {set_i + 1} Best Fitness')
        axs[ax_row, ax_col].legend()

        axs[ax_row, ax_col].plot(set_generations[set_i], set_average_fitness[set_i], label='Average Fitness')
        axs[ax_row, ax_col].set_title(f'Set {set_i + 1} Average Fitness')
        axs[ax_row, ax_col].legend()
    plt.tight_layout()
    plt.savefig(f'set_{figure_i//9+1}_figure.png')  # Save the figure with a unique name
    plt.show()
