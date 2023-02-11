import numpy as np
import random
import gym
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input

# Turn off tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def save_agent(model, env, generation, version):
    """ Save model function """
    print('Saving model...')
    save_model(model, 'Training/Saved Models/{}_{}_{}.h5'.format(env, generation, version))


## Is this different to the Flatten function from Tensorflow?
def flatten(self,individual):
        """ Mutation """
        # Flatten weights
        flattened_weights = individual.model.get_weights()
        flattened_weights = [w.flatten() for w in flattened_weights]
        flattened_weights = np.concatenate(flattened_weights)
        return flattened_weights


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
            raise ValueError('Environment doesn"t quite work yet. Reward is always < 0 and is too random.')
            agentPopulation = [NeuralNetwork(2, 3) for _ in range(self.population_size)]                            
        elif env == 'CartPole-v1':
            agentPopulation = [NeuralNetwork(4, 2) for _ in range(self.population_size)]
        else:
            raise ValueError('Environment not supported')
        
        self.population = agentPopulation


    def fitness(self):
        """ 
        This function evaluates the fitness of the population. The fitness is currently the sum of rewards.
        Each individual is loaded and passed through the environment 
        :param individual: individual neural network
        """
        # Initialize population variables
        max_fitness = 0
        max_individual = None
        self.population_fitness = []

        # Iterate through the population
        for index, individual in enumerate(self.population):
            print('Status: {}/{}'.format(index, len(self.population)))
            
            # Create a new environment and initialize 
            env = gym.make(self.environment)
            observation = env.reset()
            done = False

            # Initialize fitness score
            fitness = 0

            # Pass individual through environment. Fitness is the sum of rewards. 
            # This section can be tampered with a lot
            while not done:
                action = np.argmax(individual.model.predict(observation.reshape(1, -1)))
                observation, reward, done, info = env.step(action)
                fitness += reward

            # Print fitness score
            print('Fitness: {}'.format(fitness))

            # Update max fitness and individual
            if fitness > max_fitness:
                max_fitness = fitness
                max_individual = individual

            # create a list of fitness values for the entire population
            self.population_fitness.append(fitness)

        return max_individual, max_fitness


    def selection(self, selection_type):
        ''' 
        Function to select the best individuals from the population. 



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
                        selected_population.append(index)

                    return selected_population

            selected_population = selection(self)

            # a list of indices of the selected individuals from the original population
            return selected_population

        
        elif selection_type == 'tournament':
            """
            Tournament selection
            Selects the best individual from a random sample of individuals.
            """
            # chosen size to be around 20% of the population - can be changed
            def selection(self, t_size=len(self.population)//5):
                # selecting a random sample of indexes of individuals 
                selected_individuals = random.sample(range(len(self.population)), t_size)

                # selecting the best individual from the random sample
                selected_individual = max(selected_individuals, key=lambda x: self.population_fitness[x])

                return selected_individual
            
            for _ in range(len(self.population)):
                selected_population.append(selection(self))

            # a list of indices of the selected individuals from the original population
            return selected_population


    def crossover(self, selected_population):
        """ Crossover
        input: index for selected agents for crossover
        output: arrays of weights for crossed over agents
        
         """
        offspring1 = []
        offspring2 = []
       
       # This section loops through whole population and selects two parents at random, and does for every pair
        # for i in range(len(selected_population)//2):
        #     parent1 = random.randint(0,len(selected_population)-1)
        #     parent2 = random.randint(0,len(selected_population)-1)
        #     if parent1 == parent2:
        #         parent2 = random.randint(0,len(selected_population)-1)

        # This selects two parents at random from the selected population
        parent1 = np.random.choice(selected_population, 1, replace=False)
        parent2 = np.random.choice(selected_population, 1, replace=False)
        parent1 = self.population[parent1]
        parent2 = self.population[parent2]

        # This flattens the weights of the parents
        parent1 = self.flatten(parent1)
        parent2 = self.flatten(parent2)

        # This selects a point at random to split the parents
        split = random.ragendint(0,len(parent1)-1)
        # This creates the children by combining the parents
        child1_genes = np.array(parent1[0:split].tolist() + parent2[split:].tolist())
        child2_genes = np.array(parent2[0:split].tolist() + parent1[split:].tolist())
                
        # append the children to the offspring list
        offspring1.append(child1_genes)
        offspring2.append(child2_genes)
            # agents.extend(offspring)
            # return agents
        # pass
        return offspring1, offspring2


    def mutate(self,flattened_weights):
        """ Mutation """
        # Mutate weights
        for i in range(len(flattened_weights)):
            flattened_weights[i] *= 1+(random.uniform(-self.mutation_rate, self.mutation_rate))
        return flattened_weights
        

    def run(self, num_generations):
        """ Run the genetic algorithm 
        :param num_generations: number of generations
        """
        gen = 0
        while gen < num_generations:
            # Initialize population
            self.init_population()

            # Evaluate fitness
            max_individual, max_fitness = self.fitness()
            
            # max_individual.model.save('best_model.h5')  
            # selected_population = self.selection('roulette_wheel')
            


            # Perform crossover
            # offspring1, offspring2 = self.crossover(selected_population)
            save_agent(max_individual.model, env, '1', 'v1')
            gen += 1




if __name__ == "__main__":
    
    # Create environment
    env = 'CartPole-v1'

    # Create genetic algorithm
    ga = GeneticAlgorithm(1, 0.1, 0.7, env)

    # Run genetic algorithm
    ga.run(1)

