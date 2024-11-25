import numpy as np
import random
from PIL import Image
import time

TIME = time.time()
TARGET = None

def img_to_gene(image: Image):
    """converts the given PIL Image object to a NumPy array of RGB values

    Args:
        image (Image): the image object to be converted

    Returns:
        Array: an array of RGB tuples, representing pixels in the original image
    """
    return np.array(image).reshape(-1, 3)

def gene_to_img(array: np.ndarray, dimensions):
    """converts the given NumPy array of RGB values to a PIL Image

    Args:
        array (np.ndarray): an array of RGB tuples
        dimensions (tuple): dimensions reshaping the array

    Returns:
        Image: PIL Image constructed from the given Array
    """
    img_array = array.reshape(*dimensions[::-1], -1)
    return Image.fromarray(img_array.astype('uint8'), 'RGB')

def generate_initial_population(pop_size, num_pixels):
    """Creates pop_size number of arrays, each num_pixels long, with randomized values in the RGB tuple

    Args:
        pop_size (int): number of arrays to be created
        num_pixels (int): length of each array

    Returns:
        Array: an array of arrays representing random noisy images
    """
    return np.random.randint(0, 256, size=(pop_size, num_pixels, 3))

def calculate_fitness(gene):
    """determines the MSE between the given gene and the target gene

    Args:
        gene (Array): an array representing a candidate image

    Returns:
        float: the MSE between the target and test values, normalized between 1 and 0
    """
    mse = np.mean((gene - TARGET) ** 2)
    return 1 / (mse + 1e-10)

def calculate_fitnesses(population):
    """calculates the fitness of each member of the population, and returns the population sorted by fitness

    Args:
        population (array): an array of arrays representing a member of the population

    Returns:
        float, array: returns the best fitness value, and the sorted population array
    """
    fitness_values = np.array([calculate_fitness(row) for row in population])
    sorted_indices = np.argsort(fitness_values)[::-1]
    return fitness_values[sorted_indices[0]], population[sorted_indices]

def select_survivors(population, num_survivors):
    """selects which members of the population get to survive

    Args:
        population (array): an array of arrays representing a candidate image
        num_survivors (int): number of survivors that get to reproduce

    Returns:
        array: a slice of the population who get to reproduce
    """
    return population[-num_survivors:]

def create_new_population(population, pop_size, mutation_rate):
    """creates the new generation from the survivors

    Args:
        population (array): array of arrays representing a candidate image
        pop_size (int): the size the population needs to be
        mutation_rate (float): the rate individual pixels should mutate during reproduction

    Returns:
        array: the new generation array
    """
    num_elites = len(population)
    num_offspring = pop_size - num_elites
    
    parents = get_parents(population, num_offspring // 2)
    offspring = crossover(parents)
    offspring = mutate(offspring, mutation_rate)
    
    return np.vstack((population, offspring))

def get_parents(population, num_pairs):
    """creates a list of 2-tuples, each representing a reproductive pair

    Args:
        population (array): array of arrays representing candidate images
        num_pairs (int): number of pairs to create

    Returns:
        List: a list of reproduction pairs, stored as 2-tuples
    """
    n = len(population)
    pairs = np.random.choice(n, size=(num_pairs, 2), replace=True)
    return [(population[i], population[j]) for i, j in pairs]

def crossover(parents):
    """creates an offspring image through 1 point crossover (selecting a random point in a parent and replacing everything after that with the other parent)

    Args:
        parents (tuple): two arrays representing parent images

    Returns:
        array: an array representing an offspring image
    """
    offspring = []
    
    for parent1, parent2 in parents:
        crossover_point = random.randint(1, parent1.shape[0] - 1)
        child = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)
        
    return np.array(offspring)

def mutate(offspring, mutation_rate):
    """chooses random points in the given candidate array, and replaces them with random RGB values

    Args:
        offspring (array): an array representing a candidate image
        mutation_rate (float): the percentage of pixels to be mutated

    Returns:
        array: candidate array with randomized values
    """
    mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
    mutation_values = np.random.randint(-10, 10, size=offspring.shape)
    offspring[mutation_mask] += mutation_values[mutation_mask]
    np.clip(offspring, 0, 255, out=offspring)
    return offspring

def print_details(fitness, genome, generation, generations, dimensions):
    """Prints a summary string about the current generation, and saves the best candidate every time the generation is divisible by 500"""
    
    print(f"Generation {generation}/{generations}")
    print(f"Best Fitness: {fitness}")
    print(f"Time taken: {round(time.time() - TIME, 2)} seconds")
    if generation % 500 == 0 or generation == generations - 1:
        img = gene_to_img(genome, dimensions)
        img.save(f"image-reproduction/images/generation-{generation}.png")

def reconstruct_image(target="image-reproduction/target.jpg", generations= 10000, pop_size=100, num_survivors=20, mutation_rate = 0.05):
    global TIME, TARGET
    target_image = Image.open(target)
    TARGET = img_to_gene(target_image)
    
    population = generate_initial_population(pop_size, TARGET.shape[0])
    for generation in range(generations):
        TIME = time.time()
        best_fitness, population = calculate_fitnesses(population)
        
        population = select_survivors(population, num_survivors)
        
        if generation % 1000 == 0:
            print_details(best_fitness, population[0], generation, generations, target_image.size)
        
        population = create_new_population(population, pop_size, mutation_rate)

if __name__ == "__main__":
    reconstruct_image()
