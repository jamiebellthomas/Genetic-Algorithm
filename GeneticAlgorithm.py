import numpy as np

import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input

class NeuralNetwork():
    """ Neural Network class """
    def __init__(self, input_size, output_size):
        """ Constructor 
        :param input_size: size of the input layer
        :param output_size: size of the output layer
        """
        # Input layer with input_size nodes, dense layer with 5 nodes and output layer with output_size nodes
        input_layer  = Input(input_size)
        dense_layer1 = Dense(5, activation="relu")
        output_layer = Dense(output_size, activation="linear")
        
        # Assign layers to the model
        model = Sequential()
        model.add(input_layer)
        model.add(dense_layer1)
        model.add(output_layer)

        # Create random weights
        weights = model.get_weights()
        weights = [np.random.rand(*w.shape) for w in weights]
        model.set_weights(weights)

        self.model = model




class GeneticAlgorithm():
    """ Genetic Algorithm class """
    def __init__(self, population_size, mutation_rate, crossover_rate, environment):
        """ Constructor 
        :param population_size: size of the population
        :param mutation_rate: mutation rate
        :param crossover_rate: crossover rate
        :param environment: Gym environment
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.environment = environment

    def init_population(self):
        """ Initialize population
        Create a population of neural networks with random weights.
         """
        env = self.environment

        if env == 'MountainCar-v0':
            agentPopulation = [NeuralNetwork(2, 3) for _ in range(self.population_size)]                            
        else:
            raise ValueError('Environment not supported')
        
        self.population = agentPopulation


    def fitness(self, population):
        """ Fitness function 
        :param individual: individual neural network
        """
        max_fitness = 0
        for index, individual in enumerate(population):
            print('Status: {}/{}'.format(index, len(population)))
            env = self.environment
            env = gym.make(env)
            observation = env.reset()
            done = False
            fitness = 0

            while not done:
                action = np.argmax(individual.model.predict(observation.reshape(1, -1)))
                observation, reward, done, info = env.step(action)
                fitness += reward

            if fitness > max_fitness:
                max_fitness = fitness
                max_individual = individual

        return max_individual, max_fitness


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
        """ Run the genetic algorithm 
        :param num_generations: number of generations
        """
        gen = 0
        while gen < num_generations:
            # Initialize population
            self.init_population()

            # Evaluate fitness
            max_individual, max_fitness = self.fitness(self.population)
            
            # Perform selection

            # Perform crossover

            # Perform mutation
            ga.max_individual = self.mutation(ga.max_individual)
            gen += 1

        



if __name__ == "__main__":
    
    # Create environment
    env = 'MountainCar-v0'

    # Create genetic algorithm
    ga = GeneticAlgorithm(10, 0.1, 0.7, env)

    # Run genetic algorithm
    ga.run(1)