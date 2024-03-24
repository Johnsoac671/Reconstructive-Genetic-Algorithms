from itertools import combinations
from PIL import Image
import random
import numpy as np
import time

TIME = 0
class Genome:
    def __init__(self, dimensions, image=None):
        
        if image == None:
            self.image = Image.new("RGB", dimensions)
        else:
            self.image = image
            
        self.width = dimensions[0]
        self.height = dimensions[1]
        
        self.fitness = -1
        
    def create_random_image(self):
        random_colors = np.random.randint(0, 256, size=(self.height, self.width, 3), dtype=np.uint8)
        self.image = Image.fromarray(random_colors, mode='RGB')
                
    def calculate_fitness(self, target: Image):
        if self.fitness != -1:
            return
        
        target_array = np.array(target)
        image_array = np.array(self.image)
        
        total_abs_diff = np.sum(np.abs(target_array - image_array))
        
        
        max_diff = 255 * 3 * image_array.size
        
        
        self.fitness = round(1 - (total_abs_diff / max_diff), 4)
        
                

    def save(self, id=random.randint(0, 1000000)):
        self.image.save(f"genome-{id}.png")
    
    def close(self):
        self.image.close()

def crossover(parent1: Genome, parent2: Genome, crossover_chance=0.5):
    image2 = np.array(parent2.image)
    
    offspring = np.array(parent1.image)
    height, width, _ = offspring.shape
    
    for row in range(height):
        for col in range(width):
            if random.random() > crossover_chance:
                offspring[row][col] = image2[row][col]
                
    return Image.fromarray(offspring, mode="RGB")

def mutate(genome: Genome, sigma):
    image = np.array(genome.image).astype(np.float32)
    noise = np.random.normal(0, sigma, size=image.shape)
    
    image += noise
    image = np.clip(image, 0, 255)
    genome.image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    
    
    return genome

def selection_process(genomes, k):
    selected = genomes[:len(genomes) // 4]
    genomes = genomes[(len(genomes) // 4):]

    random.shuffle(genomes)
    
    while len(genomes) > 0:
        
        if len(genomes) <= k:
            tournament = genomes.copy()
            genomes = []
        else:
            tournament = genomes[:k]      
            genomes = genomes[k:]

            
        selected.append(max(tournament, key=lambda x: x.fitness))
        
    return selected

def display_statistics(genome, generation):
    global TIME
    print(f"Generation {generation}")
    print(f"Best Fitness: {genome.fitness}")
    print(f"Completed in {round(time.time() - TIME, 2)} seconds")
    TIME = time.time()
    print()
    if generation % 3 == 0:
        genome.save(generation)
    



    
def reconstruct_image(target="target.png", **kwargs):
    global TIME
    target_image = Image.open(target)
    generations = kwargs.get("generations", 10)
    generation_size = kwargs.get("generation_size", 50)
    
    update_stats = kwargs.get("update_stats", generations // 1)
    
    # mutation_rate = kwargs.get("mutation_rate", 0.1)
    sigma = kwargs.get("sigma", 0.013)
    
    tournament_size = kwargs.get("tournament_size", 3)
    
    genomes = [Genome(target_image.size) for _ in range(generation_size)]
    TIME = time.time()
    for genome in genomes:
        genome.create_random_image()

    for generation in range(1, generations+1):

        for genome in genomes:
            genome.calculate_fitness(target_image)
        
        genomes.sort(key=lambda x: x.fitness, reverse=True)
        
        if generation % update_stats == 0:
            display_statistics(genomes[0], generation)
        
        genomes = selection_process(genomes, tournament_size)
        
        parents = list(combinations(genomes, 2))
        
        random.shuffle(parents)
        parents = parents[:generation_size - len(genomes)]

        for couple in parents:
            offspring = Genome(target_image.size, crossover(*couple))
            mutate(offspring, sigma)
            genomes.append(offspring)
            
    
    display_statistics(genomes[0], generation)
    
if __name__ == "__main__":
    reconstruct_image()
