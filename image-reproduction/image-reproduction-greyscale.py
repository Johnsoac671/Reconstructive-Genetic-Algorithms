import numpy as np
import random
from PIL import Image
import time

TIME = time.time()
TARGET = None

def img_to_gene(image: Image):
    image_array = np.array(image)
    return image_array.flatten()

def gene_to_img(array: np.ndarray, dimensions):
    img_array = array.reshape(dimensions[::-1])
    img = Image.fromarray(img_array.astype('uint8'), 'L')
    return img

def generate_initial_population(pop_size, num_pixels):
    random_pop = np.random.randint(0, 256, size=(pop_size, num_pixels))
    return random_pop

def calculate_fitness(gene):
    return -1 / np.sum(np.abs(gene - TARGET))

def calculate_fitnesses(population: np.ndarray):
    global TARGET
    fitness_values = np.array([calculate_fitness(row) for row in population])

    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = population[sorted_indices]

    highest_fitness_value = fitness_values[sorted_indices[0]]

    return highest_fitness_value, sorted_population

def select_survivors(population, num_survivors):
    n = len(population)
    selection_probabilities = np.linspace(1, 0, n)
    selection_probabilities /= selection_probabilities.sum()

    selected_indices = np.random.choice(np.arange(n), size=num_survivors, replace=True, p=selection_probabilities)

    return population[selected_indices]

def create_new_population(population, pop_size, mutation_rate):
    elites = population[:(len(population) // 5)]
    parents = get_parents(population, pop_size - (len(population) // 10))
    offspring = crossover(parents)
    offspring = mutate(offspring, mutation_rate)
    return np.vstack((elites, offspring))

def get_parents(population, num_parents):
    n = len(population)
    num_pairs_needed = num_parents // 2

    random_indices = np.random.choice(n, size=(num_pairs_needed, 2), replace=True)
    
    parents = [(population[i], population[j]) for i, j in random_indices]
    
    return parents
    
def crossover(parents):
    offspring = []
    
    for parent1, parent2 in parents:
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)
        
    return np.array(offspring)

def mutate(offspring, mutation_rate):
    mutated_offspring = offspring.copy() 
    

    mask = np.random.random(size=offspring.shape) < mutation_rate
    

    mutation_values = np.random.normal(loc=0, scale=50, size=offspring.shape).astype(np.int32)
    

    mutated_offspring += mutation_values * mask
    

    mutated_offspring = np.clip(mutated_offspring, 0, 255, out=mutated_offspring)
    
    return mutated_offspring

def print_details(fitness, genome, generation, generations, dimensions):
    global TIME
    print(f"Generation {generation}/{generations}")
    print(f"Best Fitness: {fitness}")
    print(f"Time taken: {round(time.time() - TIME,2)}")
    TIME = time.time()
    if generation % 1000 == 0:
        gene_to_img(genome, dimensions).save(f"image-reproduction/images/generation-{generation}.png")

def reconstruct_image(target="image-reproduction/target.jpg"):
    global TIME
    global TARGET
    target_image = Image.open(target).convert('L')
    TARGET = img_to_gene(target_image)
    
    generations = 100000
    genomes = 100
    num_survivors = 10
    
    population = generate_initial_population(genomes, TARGET.shape[0])
    for generation in range(generations):
        best_fitness, population = calculate_fitnesses(population)
        
        if generation % 10 == 0:
            print_details(best_fitness, population[0], generation, generations, target_image.size)
        
        surviving_population = select_survivors(population, num_survivors)
        
        population = create_new_population(surviving_population, genomes, 0.01)

reconstruct_image()
