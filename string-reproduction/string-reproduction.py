import random
import string
import time
from jellyfish import hamming_distance

CHARACTERS = string.ascii_lowercase + string.ascii_uppercase + string.punctuation + string.digits + " "

def generate_random_pop(pop_size, target_length):
    population = [''.join(random.choices(CHARACTERS, k=target_length)) for _ in range(pop_size)]
    
    return population


def calculate_pop_fitnesses(population, target):
    pop_with_fitness = [(calculate_fitness(x, target), x) for x in population]
    return sorted(pop_with_fitness, key=lambda x: x[0], reverse=True)


def calculate_fitness(string, target):
    return round(1 - (hamming_distance(string, target) / len(target)), 2)


def display_statistics(generation, generations, best_candidate, start_time):
    print(f"Generation: {generation}/{generations}")
    print(f"Completed in: {round(time.time() - start_time, 3)}")
    print(f"Best String: {best_candidate[1]}")
    print(f"Fitness: {best_candidate[0]}")
    print()


def normalize_fitness(population):
    total_fitness = sum(individual[0] for individual in population)
    
    for individual in population:
        individual = (individual[0] / total_fitness, individual[1])
    
    return population


def selection(population, num_parents):
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
    n = len(pop)
    pairs = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(num_pairs_needed)]
    
    return [(pop[i], pop[j]) for i, j in pairs]


def create_new_generation(population, pop_size, elite_ratio, mutation_rate):
    num_elites = int(len(population) * elite_ratio)
    next_gen = population[:num_elites]
    
    parents = get_parents(population, pop_size - num_elites)
    
    for pair in parents:
        child = crossover(*pair)
        child = mutate(child, mutation_rate)
        next_gen.append(child)
    
    
    return next_gen


def crossover(parent1, parent2):
    random_point = random.randint(1, len(parent1) - 1)
    
    if random.random() > 0.5:
        return parent1[:random_point] + parent2[random_point:]
    else:
        return parent2[:random_point] + parent1[random_point:]
    


def mutate(string, mutation_rate):
    
    new_string = ""
    
    for char in string:
        if random.random() <= mutation_rate:
            new_string += random.choice(CHARACTERS)
        else:
            new_string += char
        
    return new_string
    


def reproduce_string(target, generation_size, generations, elitism_ratio, mutation_rate):
    
    population = generate_random_pop(generation_size, len(target))
    
    for generation in range(generations):
        start = time.time()
        population = calculate_pop_fitnesses(population, target)
         
        if generation % 10 == 0:
            display_statistics(generation, generations, population[0], start)
            
        if population[0][0] == 1.0:
            print(f"Target String: {target}")
            print(f"Reconstructed String: {population[0][1]}")
            print()
            break
        
        population = selection(population, 100)
        
        population = create_new_generation(population, generation_size, elitism_ratio, mutation_rate)


reproduce_string("Now this is a story, all about how, my life got flipped, turned upside down!", 1000, 1000, 0.2, 0.01)