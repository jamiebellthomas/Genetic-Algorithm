import numpy as np

class GeneticAlgorithm():
    """ Genetic Algorithm class """
    def __init__(self, population_size, mutation_rate, crossover_rate):
        """ Constructor 
        :param population_size: size of the population
        :param mutation_rate: mutation rate
        :param crossover_rate: crossover rate
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def init_population(self):
        """ Initialize population """
        pass

    def fitness(self, individual):
        """ Fitness function """
        pass

    def selection(self):
        """ Selection """
        pass

    def crossover(self, parent1, parent2):
        """ Crossover """
        pass

    def mutation(self, individual):
        """ Mutation """
        pass

    def run(self, num_generations):
        """ Run the genetic algorithm """
        
        # do loop for num_generations
            # Initialize population

            # Evaluate fitness

            # Perform selection

            # Perform crossover

            # Perform mutation

        # Return best individual