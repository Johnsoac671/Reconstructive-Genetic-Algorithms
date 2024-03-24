
from PIL import Image
import random
import numpy as np

class Genome:
    def __init__(self, dimensions, id, image=None):
        
        if image == None:
            self.image = Image.new("RGB", dimensions)
        else:
            self.image = image
            
        self.width = dimensions[0]
        self.height = dimensions[1]
        
        self.id = id
        
        self.fitness = 0.0
        
    def create_random_image(self):
        random_colors = np.random.randint(0, 256, size=(self.height, self.width, 3), dtype=np.uint8)
        self.image = Image.fromarray(random_colors, mode='RGB')
                
    def calculate_fitness(self, target):
        target_array = np.array(target)
        image_array = np.array(self.image)
        
        total = np.sum(np.abs(target_array - image_array))
        
        try:
            self.fitness = 1 / (total / (255 * 3) * (image_array.size))
        except ZeroDivisionError:
            self.fitness = 1
                

    def save(self):
        self.image.save(f"genome-{self.id}.png")
    
    def close(self):
        self.image.close()

def crossover(parent1: Genome, parent2: Genome, crossover_chance=0.5):
    image2 = np.array(parent2.image)
    
    offspring = np.array(parent1.image)
    
    for col in range(len(offspring)):
        for row in range(len(col)):
            if random.random > crossover_chance:
                offspring[col][row] = image2[col][row]
                
    return Image.fromarray(offspring, mode="RGB")

def mutate(genome: Genome, sigma):
    image = np.array(genome.image)
    noise = np.random.normal(0, sigma, size=image.shape)
    
    image += noise
    
    genome.image = Image.fromarray(np.clip(image, 0, 255))
    
    
    return genome

       
