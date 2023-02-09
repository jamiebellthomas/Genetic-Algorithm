import numpy as np
import random
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

    def flatten(self,individual):
        """ Mutation """
        # Flatten weights
        flattened_weights = individual.model.get_weights()
        flattened_weights = [w.flatten() for w in flattened_weights]
        flattened_weights = np.concatenate(flattened_weights)
        return flattened_weights

    def crossover(self, parent1, parent2):
        """ Crossover
        input: index of arrays required for crossover
        output: arrays of crossover children
        
         """
        offspring1 = []
        offspring2 = []
        # selected_population = self.selection('roulette_wheel')
        # for i in selected_population
        #     parent1 = self.population[parent1].model.get_weights()
        #     parent2 = self.population[parent2].model.get_weights()
        
        parent1 = self.flatten(parent1)
        parent2 = self.flatten(parent2)
        print(parent1)
        print(parent2)

        split = random.randint(0,len(parent1)-1)
        child1_genes = np.array(parent1[0:split].tolist() + parent2[split:].tolist())
        child2_genes = np.array(parent2[0:split].tolist() + parent1[split:].tolist())
                
        # child1.neural_network.weights = unflatten(child1_genes,shapes)
        # child2.neural_network.weights = unflatten(child2_genes,shapes)
        
        offspring1.append(child1_genes)
        print(offspring1)
        offspring2.append(child2_genes)
            # agents.extend(offspring)
            # return agents
        # pass
        return offspring1, offspring2


    def run(self, num_generations):
        """ Run the genetic algorithm 
        :param num_generations: number of generations
            """

        self.init_population()
        # print the weights for each individual in the population
        # for individual in self.population:
        #     print(individual.model.get_weights())

        # gen = 0
        # while gen < num_generations:
        #     # Initialize population
        #     self.init_population()

            # Evaluate fitness
            # max_individual, max_fitness = self.fitness(self.population)
            
            # Perform selection

            # Perform crossover
        # flattened_weights = self.flatten(self.population[0])
        # print(flattened_weights)
            
        offspring1, offspring2 = self.crossover(self.population[0], self.population[1])
            # Perform mutation
            # ga.max_individual = self.mutation(ga.max_individual)
            # gen += 1

        print('the difference between the two offspring is: ', np.subtract(offspring1, offspring2))


if __name__ == "__main__":
    
    # Create environment
    env = 'MountainCar-v0'

    # Create genetic algorithm
    ga = GeneticAlgorithm(10, 0.1, 0.7, env)

    # Run genetic algorithm
    ga.run(1)

