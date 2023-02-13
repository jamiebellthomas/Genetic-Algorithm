import numpy as np
import random
import gym

from selection import selection
from NeuralNetwork import NeuralNetwork
from saving_data import save_generation




class GeneticAlgorithm():
    """ Genetic Algorithm class """
    def __init__(self, population_size, mutation_rate, crossover_rate, environment, description):
        """ Constructor 
        
        parameters:
        ----------------
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


    def evaluate_fitness(self):
        """ 
        This function evaluates the fitness of the population. The fitness is currently the sum of rewards.
        Each agent is loaded and passed through the environment 
        
        parameters:
        ----------------
            None
                Population is taken as a class variable

        returns:
        ----------------
            population_fitness: list <float>
                List of fitness scores for each agent in the population
        """
        # Initialize population fitness list.
        population_fitness = []

        # Iterate through the population
        for index, agent in enumerate(self.population):
            print('Status: {}/{}'.format(index, len(self.population)))
            
            # Create a new environment and initialize variables
            env = gym.make(self.environment)
            observation = env.reset()
            done = False

            # Initialize fitness score
            fitness = 0

            # Pass agent through environment. Fitness is the sum of rewards. 
            # This section can be tampered with a lot
            while not done:
                thoughts = agent.model.predict(observation.reshape(1, -1))
                print(thoughts)
                action = np.argmax(thoughts)
                observation, reward, done, info = env.step(action)
                fitness += reward

            # Print fitness score
            print('Fitness: {}'.format(fitness))
            population_fitness.append(fitness)

        # Return the population fitness
        return population_fitness


    
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
        This function performs crossover on the selected population.
        
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
    
       
       # This section loops through whole population and selects two parents at random, and does for every pair
        # for i in range(len(selected_population)//2):
        #     parent1 = random.randint(0,len(selected_population)-1)
        #     parent2 = random.randint(0,len(selected_population)-1)
        #     if parent1 == parent2:
        #         parent2 = random.randint(0,len(selected_population)-1)

        # for i in range(len(self.population)//2):

        # This selects the integers for the indexing of two parents at random from the selected population
        parent1 = np.random.choice(selected_population, 1, replace=False)
        parent2 = np.random.choice(selected_population, 1, replace=False)
        # This returns the neural network for the parents
        parent1 = self.population[parent1]
        parent2 = self.population[parent2]

        # This flattens the weights of the parents from a nn to a np array
        parent1 = self.flatten_nn(parent1)
        parent2 = self.flatten_nn(parent2)
        

        # This selects a point at random to split the parents
        split = random.ragendint(0,len(parent1)-1)
        # This creates the children by combining the parents
        child1_genes = np.array(parent1[0:split].tolist() + parent2[split:].tolist())
        child2_genes = np.array(parent2[0:split].tolist() + parent1[split:].tolist())
                
        # append the children to the offspring list
        offspring.append(child1_genes)
        offspring.append(child2_genes)
            # agents.extend(offspring)
            # return agents
        # pass
        return offspring



    def mutate(self,flattened_weights:np.ndarray):
        """ 
        This function mutates the weights of a given agent by a random amount between -mutation_rate and mutation_rate

        parameters: 
            column vector of weights for a given agent's neural network
        returns: 
            mutated weights for the agent
        """
        
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

            # Evaluate fitness
            population_fitness = self.evaluate_fitness()
            
            selected_population = selection(self, 'tournament', population_fitness, num_agents=2)
            print(selected_population)

            save_generation(self.population, self.description)
            # Perform crossover
            # offspring1, offspring2 = self.crossover(selected_population)
            # save_agent(max_agent.model, env, '1', 'v1')
            gen += 1




if __name__ == "__main__":
    
    # Create environment
    env = 'CartPole-v1'

    # Create genetic algorithm
    ga = GeneticAlgorithm(3, 0.1, 0.7, env, 'test')

    # Run genetic algorithm
    ga.run(1)

