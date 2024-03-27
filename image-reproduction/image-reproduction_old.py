from itertools import combinations
from PIL import Image, ImageEnhance, ImageFilter
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
        
        if target_array.shape == 2:
            total_abs_diff = np.sum(np.abs(image_array - target_array[:, :, None]))
        else:
            total_abs_diff = np.sum(np.abs(image_array - target_array))
        
        
        max_diff = 255 * 3 * image_array.size
        
        
        self.fitness = round(1 - (total_abs_diff / max_diff), 4)
        
    def save(self, id=random.randint(0, 1000000)):
        self.image.save(f"images/genome-{id}.png")
    
    def close(self):
        self.image.close()        
        
def crossover(parent1: Genome, parent2: Genome, crossover_chance=0.5):
    image1 = np.array(parent1.image)
    image2 = np.array(parent2.image)
    
    mask = np.random.rand(*image1.shape[:2]) <= crossover_chance
    offspring = np.where(mask[:, :, np.newaxis], image1, image2)
    
    return Image.fromarray(offspring.astype(np.uint8), mode="RGB")


def mutate(genome: Genome, mutation_rate):
    image_array = np.array(genome.image)
    
    mask = np.random.rand(*image_array.shape[:2]) < mutation_rate
    
    new_colors = np.random.randint(0, 256, size=(np.sum(mask), 3))
    
    image_array[mask] = new_colors
    
    mutated_image = Image.fromarray(image_array)
    genome.image = mutated_image
    
    return genome


def selection_process(genomes, k):
    selected = genomes[:len(genomes) // 5]
    genomes = genomes[(len(genomes) // 5):]

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
    
    genome.save(generation)
    
def reconstruct_image(target="target.jpg", **kwargs):
    global TIME
    
    target_image = Image.open(target)
    generations = kwargs.get("generations", 20000)
    generation_size = kwargs.get("generation_size", 100)
    mutation_rate = 0.1
    update_stats = kwargs.get("update_stats", generations // 100)
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
            mutate(offspring, mutation_rate)
            genomes.append(offspring)
            
            
            
    
    display_statistics(genomes[0], generation)
    
if __name__ == "__main__":
    reconstruct_image()
