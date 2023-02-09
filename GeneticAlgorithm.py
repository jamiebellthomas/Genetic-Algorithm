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
        # Sets random weights to the model's weights and biases
        weights = model.get_weights()
        weights = [np.random.rand(*w.shape) for w in weights]
        model.set_weights(weights)

        self.model = model
        self.layers = model.layers



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

            # initialize a list of fitness values for the entire population
            self.population_fitness = []

            while not done:
                action = np.argmax(individual.model.predict(observation.reshape(1, -1)))
                observation, reward, done, info = env.step(action)
                fitness += reward

            if fitness > max_fitness:
                max_fitness = fitness
                max_individual = individual

            # create a list of fitness values for the entire population
            self.population_fitness.append(fitness)

        return max_individual, max_fitness

    def flatten(self,individual):
        """ Mutation """
        # Flatten weights
        flattened_weights = individual.model.get_weights()
        flattened_weights = [w.flatten() for w in flattened_weights]
        flattened_weights = np.concatenate(flattened_weights)
        return flattened_weights

    def selection(self, selection_type):
        ''' 
            :param selection_type: type of selection
            :param self.population: population of neural networks
            :param self.population_fitness: fitness values of the population

            :return: selected_population
            :return: selected_individual
        '''
        # initializing a new population
        selected_population = []

        if selection_type == 'roulette_wheel':
            """
            Roulette wheel selection 
            The probability of selecting an individual is proportional to its fitness value.
            """
            # computing probabilities for each individual
            fitness_sum = sum(self.population_fitness)
            self.selection_probs = [fitness / fitness_sum for fitness in self.population_fitness]

            # generating a random number between 0 and 1 for the population selection
            random_number = np.random.rand()

            # selecting the population based on the random number
            def selection(self, random_number=random_number):
                for index, prob in enumerate(self.selection_probs):
                    random_number -= prob
                    # all the individuals with a probability greater than the random number are selected
                    if random_number <= 0:
                        selected_population.append(self.population[index])

                    return selected_population

            selected_population = selection(self)

            return selected_population

        
        elif selection_type == 'tournament':
            """
            Tournament selection
            Selects the best individual from a random sample of individuals.
            """
            def selection(self, t_size=len(self.population)//5):
                # selecting a random sample of individuals
                sample = np.random.choice(self.population, t_size, replace=False)
                # selecting the best individual from the sample
                selected_individual = max(sample, key=lambda x: x.fitness)
                selected_population.append(selected_individual)
                return selected_population
            
            for _ in range(len(self.population)):
                selected_population = selection(self)

            return selected_population


    def crossover(self, parent1, parent2):
        """ Crossover """
        pass


    def mutate(self,flattened_weights):
        """ Mutation """
        # Mutate weights
        for i in range(len(flattened_weights)):
            flattened_weights[i] *= 1+(random.uniform(-self.mutation_rate, self.mutation_rate))
        return flattened_weights
        

        
        
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
