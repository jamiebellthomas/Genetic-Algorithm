import numpy as np
import random
import gym

from selection import selection
from NeuralNetwork import NeuralNetwork
from saving_data import save_generation
from evaluate_fitness import evaluate_fitness
from MutationFlattenUnflatten import flatten, unflatten, mutate


def flatten_nn(agent):
    """ Mutation 
    """
    # Flatten weights
    flattened_weights = agent.model.get_weights()
    flattened_weights = [w.flatten() for w in flattened_weights]
    flattened_weights = np.concatenate(flattened_weights)
    return flattened_weights

class GeneticAlgorithm():
    """ Genetic Algorithm class """
    def __init__(self, num_generations, population_size, mutation_rate, crossover_rate, environment, description):
        """ Constructor 
        
        parameters:
        ----------------
            num_generations: int
                Number of generations to train for
            population_size: int
                Size of the population
            mutation_rate: float
                Mutation rate
            crossover_rate: float
                Crossover rate
            environment: str
                Environment name corresponding to the OpenAI Gym environment
            description: str
                Description of the model. This data is saved to the ModelDetails.csv file.
        """
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.environment = environment
        self.population = self.init_population(environment)
        self.description = description


    def init_population(self, env):
        """ Initialize population
        Create a population of neural networks with random weights.
        Input to the neural network corrsponds to the observation space.
        Output of the neural network corresponds to the action space.

        parameters:
        ----------------
            env: str
                Environment name corresponding to the OpenAI Gym environment

        returns:
        ----------------
            agentPopulation: list <NeuralNetwork>
                List of NeuralNetwork objects representing the population
         """
        env = self.environment

        print('Initializing population for {}...'.format(env))
        if env == 'MountainCar-v0':
            # agentPopulation = [NeuralNetwork(2, 3) for _ in range(self.population_size)]                            
            raise ValueError('Environment doesn"t quite work yet. Reward is always < 0 and is too random.')
        elif env == 'CartPole-v1':
            agentPopulation = [NeuralNetwork(4, 2) for _ in range(self.population_size)]
        else:
            raise ValueError('Environment not supported')
        
        return agentPopulation



    
    def flatten_nn(self,agent):
        """ Flatten neural network weights

        input: neural network to be flattened into np array

        ----------------

        output: flattened np array of weights for neural network
        """
        # Flatten weights
        flattened_weights = agent.model.get_weights()
        flattened_weights = [w.flatten() for w in flattened_weights]
        flattened_weights = np.concatenate(flattened_weights)
        return flattened_weights

        
    def crossover(self, selected_population):
        """ Crossover
        This function performs crossover on the selected population in as many different random way in order to fill the population back up to the original size.
        
        parameters:
        ----------------
            selected_population: list <int>
                List of indices of the selected population

        returns:
        ----------------
            offspring: list <np.array>
                Flattened weights of the offspring
         """
        offspring = []
       
        # Crosover from the selected popuplation to fill the population back up to the original size
        for i in range((self.population_size)//2):

            # This selects the integers for the indexing of two parents at random from the selected population
            parent1 = selected_population[random.choice(range(len(selected_population)))]
            parent2 = selected_population[random.choice(range(len(selected_population)))]
           # This checks if the parents are the same and if so, selects a new parent
            if parent1 == parent2:
                parent2 = selected_population[random.choice(range(len(selected_population)))]

            # This finds the neural network for the parents from the population
            parent1 = self.population[parent1]
            parent2 = self.population[parent2]

            # This flattens the weights of the parents from a nn to a np array
            parent1 = self.flatten_nn(parent1)
            parent2 = self.flatten_nn(parent2)
            

            # This selects a point at random to split the parents
            split = random.randint(0,len(parent1)-1)
            # This creates the children by selecting the first half(up to splitting point) of the first parent and 
            # the second half of the second parent and then inversely for the second child
            child1_genes = np.array(parent1[0:split].tolist() + parent2[split:].tolist())
            child2_genes = np.array(parent2[0:split].tolist() + parent1[split:].tolist())
                    
            # append the children to the offspring list
            offspring.append(child1_genes)
            offspring.append(child2_genes)
    
        # print(offspring)
        return offspring



    def mutate(self,flattened_weights:list):
        """ 
        This function mutates the weights of a given agent by a random amount between -mutation_rate and mutation_rate

        parameters: 
            list of column vectors of weights for each agent
        returns: 
            mutated neural networks for each agent
        """
        next_gen = []

        for i in range(self.population_size):   
            mutated_vector = mutate(flattened_weights[i], self.mutation_rate)
            new_network = unflatten(mutated_vector, self.population[i])  
            next_gen.append(new_network)
        
    
        return next_gen
        # Mutate weights

        

    def run(self):
        """ 
        This function runs the genetic algorithm. The population is initialized in the constructor method.
        In this function, it loops through the number of generations and performs the following steps:
            1. Evaluate fitness
            2. Selection
            3. Crossover
            4. Mutation
        
        The population is saved at the end of looping.

        Potential improvements:
            - Save generations every n interations
        """
        
        gen = 0
        while gen < self.num_generations:
            print('Generation {}'.format(gen))

            # Evaluate fitness
            population_fitness = evaluate_fitness(self)
            
            selected_population = selection(self, 'tournament', population_fitness, num_agents=2)

            # save_generation(self.population, self.description)
            # Perform crossover
            offspring = self.crossover(selected_population)

            # Perform mutation
            mutated_offspring = self.mutate(offspring)

            self.population = mutated_offspring
            gen += 1

        save_generation(self)



if __name__ == "__main__":
    

    ga = GeneticAlgorithm(
        num_generations=2,
        population_size=10,
        mutation_rate=0.1,
        crossover_rate=0.7,
        environment='CartPole-v1',
        description='Testing genetic algorithm'
    )

    # Run genetic algorithm
    ga.run()

