import random
import string
import time
from jellyfish import hamming_distance

CHARACTERS = string.ascii_lowercase + string.ascii_uppercase + string.punctuation + string.digits + " " 

def generate_random_pop(pop_size, target_length):
    """Creates a list pop_size long of target_length length random strings

    Args:
        pop_size (int): number of random strings to be created
        target_length (int): length of random strings

    Returns:
        population [String]: a list of random strings
    """
    population = [''.join(random.choices(CHARACTERS, k=target_length)) for _ in range(pop_size)]
    
    return population


def calculate_pop_fitnesses(population, target):
    """Finds the fitness value of each string, and returns a packaged (fitness, string) list

    Args:
        population ([String]): List of candidate strings
        target (String): string to be reconstructed

    Returns:
        pop_with_fitness: a list of (fitness, string) tuples, sorted descending by fitness
    """
    pop_with_fitness = [(calculate_fitness(x, target), x) for x in population]
    return sorted(pop_with_fitness, key=lambda x: x[0], reverse=True)


def calculate_fitness(string, target):
    """determines the similarity between string and target using Hamming distance

    Args:
        string (String): candidate String
        target (String): correct String

    Returns:
        fitness (float): how similar the two given strings are between [0, 1]
         
    """
    return round(1 - (hamming_distance(string, target) / len(target)), 2)


def display_statistics(generation, generations, best_candidate, start_time):
    """Prints current generation statistics

    Args:
        generation (int): current generation
        generations (int): max number of generations
        best_candidate (String): closest String to target in current generation
        start_time (float): time current generation began
    """
    print(f"Generation: {generation}/{generations}")
    print(f"Completed in: {round(time.time() - start_time, 3)}")
    print(f"Best String: {best_candidate[1]}")
    print(f"Fitness: {best_candidate[0]}")
    print()


def normalize_fitness(population):
    """modifies (fitness, String) list to have normalized fitness values

    Args:
        population [(float, String)]: current population of candidate strings and their fitness values

    Returns:
        population [(float, String)]: current population of candidate strings and their fitness values, normalized
    """
    total_fitness = sum(individual[0] for individual in population)
    
    for individual in population:
        individual = (individual[0] / total_fitness, individual[1])
    
    return population


def selection(population, num_parents):
    """Selects surviving population based of SUS (Stochastic Universal Sampling)

    Args:
        population [(float, String)]: current population of candidate strings and their fitness values, normalized
        num_parents (int): size of surviving population

    Returns:
        population [String]: surviving population of candidate strings
    """
    population = normalize_fitness(population)
    num_individuals = len(population)

    pointer_distance = 1 / num_parents

    start_point = random.uniform(0, pointer_distance)

    selected_parents = []
    current_point = start_point
    current_index = 0

    for _ in range(num_parents):
        while current_point > population[current_index][0]:
            current_point -= population[current_index][0]
            current_index = (current_index + 1) % num_individuals
        selected_parents.append(population[current_index][1])
        current_point += pointer_distance

    return selected_parents


def get_parents(pop, num_pairs_needed):
    """creates list of mating pairs

    Args:
        pop [String]: surviving population
        num_pairs_needed (int): how many pairs needed to rebuild population

    Returns:
        parents [(String, String)]: list of (String, String) tuples, representing mating pairs
    """
    n = len(pop)
    pairs = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(num_pairs_needed)]
    
    return [(pop[i], pop[j]) for i, j in pairs]


def create_new_generation(population, pop_size, elite_ratio, mutation_rate):
    """Creates the next generation from previous generations survivors

    Args:
        population ([String]): surviving population
        pop_size (int): size of next generation
        elite_ratio (float): what percentage of survivors are carried over to the next generation
        mutation_rate (float): what percentage of offsprings characters to be mutated

    Returns:
        population ([String]): the next generation for testing
    """
    num_elites = int(len(population) * elite_ratio)
    next_gen = population[:num_elites]
    
    parents = get_parents(population, pop_size - num_elites)
    
    for pair in parents:
        child = crossover(*pair)
        child = mutate(child, mutation_rate)
        next_gen.append(child)
    
    
    return next_gen


def crossover(parent1, parent2):
    """Given two parent strings, returns a string created from a section of each parent string

    Args:
        parent1 (String): a surviving string from the last generation
        parent2 (String): a surviving string from the last generation

    Returns:
        child: a string stitched together from parts of the parent strings
    """
    random_point = random.randint(1, len(parent1) - 1)
    
    if random.random() > 0.5:
        return parent1[:random_point] + parent2[random_point:]
    else:
        return parent2[:random_point] + parent1[random_point:]
    


def mutate(string, mutation_rate):
    """randomly modifies characters in the offspring based of given rate

    Args:
        string (String): offspring string
        mutation_rate (float): what percentage of characters to be changed

    Returns:
        mutated_string: offspring string with mutated characters
    """
    
    new_string = ""
    
    for char in string:
        if random.random() <= mutation_rate:
            new_string += random.choice(CHARACTERS)
        else:
            new_string += char
        
    return new_string
    


def reproduce_string(target, generation_size, generations, elitism_ratio, mutation_rate):
    """reproduces the given target string

    Args:
        target (str): The string to be reproduced
        generation_size (int): number of canidate strings in each generation
        generations (int): (maximum) number of generations 
        elitism_ratio (float): percentage of each generation to be carried over
        mutation_rate (float): chance for any particular character to randomly change during reproduction
    """
    
    population = generate_random_pop(generation_size, len(target))
    
    for generation in range(generations):
        start = time.time()
        population = calculate_pop_fitnesses(population, target)
         
        if generation % 50 == 0:
            display_statistics(generation, generations, population[0], start)
            
        if population[0][0] == 1.0:
            print(f"Target String: {target}")
            print(f"Reconstructed String: {population[0][1]}")
            print(f"Completed in {generation + 1} generations")
            print()
            break
        
        population = selection(population, 100)
        
        population = create_new_generation(population, generation_size, elitism_ratio, mutation_rate)


if __name__ == "__main__":
    reproduce_string("According to all known laws of aviation, there is no way a bee should be able to fly.",
                     generation_size=200, generations=1000, elitism_ratio=0.2, mutation_rate=0.01)