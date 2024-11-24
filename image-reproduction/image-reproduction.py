import numpy as np
import random
from PIL import Image
import time

TIME = time.time()
TARGET = None

def img_to_gene(image: Image):
    return np.array(image).reshape(-1, 3)

def gene_to_img(array: np.ndarray, dimensions):
    img_array = array.reshape(*dimensions[::-1], -1)
    return Image.fromarray(img_array.astype('uint8'), 'RGB')

def generate_initial_population(pop_size, num_pixels):
    return np.random.randint(0, 256, size=(pop_size, num_pixels, 3))

def calculate_fitness(gene):
    mse = np.mean((gene - TARGET) ** 2)
    return 1 / (mse + 1e-10)

def calculate_fitnesses(population):
    fitness_values = np.array([calculate_fitness(row) for row in population])
    sorted_indices = np.argsort(fitness_values)[::-1]
    return fitness_values[sorted_indices[0]], population[sorted_indices]

def select_survivors(population, num_survivors):
    fitness_values = np.array([calculate_fitness(gene) for gene in population])
    selected_indices = np.argsort(fitness_values)[-num_survivors:]
    return population[selected_indices]

def create_new_population(population, pop_size, mutation_rate):
    num_elites = len(population)
    num_offspring = pop_size - num_elites
    
    parents = get_parents(population, num_offspring // 2)
    offspring = crossover(parents)
    offspring = mutate(offspring, mutation_rate)
    
    return np.vstack((population, offspring))

def get_parents(population, num_pairs):
    n = len(population)
    pairs = np.random.choice(n, size=(num_pairs, 2), replace=True)
    return [(population[i], population[j]) for i, j in pairs]

def crossover(parents):
    offspring = []
    for parent1, parent2 in parents:
        crossover_point = random.randint(1, parent1.shape[0] - 1)
        child = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)
    return np.array(offspring)

def mutate(offspring, mutation_rate):
    mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
    mutation_values = np.random.randint(-10, 10, size=offspring.shape)
    offspring[mutation_mask] += mutation_values[mutation_mask]
    np.clip(offspring, 0, 255, out=offspring)
    return offspring

def print_details(fitness, genome, generation, generations, dimensions):
    print(f"Generation {generation}/{generations}")
    print(f"Best Fitness: {fitness}")
    print(f"Time taken: {round(time.time() - TIME, 2)} seconds")
    if generation % 500 == 0 or generation == generations - 1:
        img = gene_to_img(genome, dimensions)
        img.save(f"image-reproduction/images/generation-{generation}.png")

def reconstruct_image(target="image-reproduction/target.jpg"):
    global TIME, TARGET
    target_image = Image.open(target)
    TARGET = img_to_gene(target_image)
    
    generations = 10000
    pop_size = 100
    num_survivors = 20
    mutation_rate = 0.05
    
    population = generate_initial_population(pop_size, TARGET.shape[0])
    for generation in range(generations):
        TIME = time.time()
        best_fitness, population = calculate_fitnesses(population)
        
        population = select_survivors(population, num_survivors)
        
        if generation % 500 == 0:
            print_details(best_fitness, population[0], generation, generations, target_image.size)
        
        population = create_new_population(population, pop_size, mutation_rate)

reconstruct_image()
