import numpy as np
import random
import gym

from selection import selection
from NeuralNetwork import NeuralNetwork
from data_manipulation import save_generation, update_model_details
from evaluate_fitness import evaluate_fitness
from mutation import mutate
from metrics import initialise_metrics, update_metrics
from crossover import crossover
from plotting_data import plot_metrics
from random_agents import random_agent


class GeneticAlgorithm():
    """ Genetic Algorithm class """
    def __init__(self, environment, population_size=5, sparse_reward=False, fitness_sharing=False, 
                num_select_agents=2, selection_type='elitism', crossover_rate=0, crossover_method='random',  
                mutation_rate=0, mutation_method='random', num_generations=2, parallel=False, plot=True,
                settings=None, description=None, save_frequency=2, random_type='fixed',initial_random_rate=0, run_tests=False, str_test_folder=None):
        """ Constructor 
        
        parameters:
        ----------------
            environment: str
                Environment name corresponding to the OpenAI Gym environment
            population_size: int
                Size of the population
            sparse_reward: bool
                Whether to use sparse reward
            fitness_sharing: bool
                Whether to use fitness sharing
            num_selected_agents: int
                Number of agents to be selected during selection
            selection_type: str
                Selection type. Values can be 'tournament', 'proportional-roulette-wheel', 
                'rank-based-rolette-wheel', 'elitism'
            crossover_rate: float
                Crossover rate
            crossover_type: str
                Crossover type. Values can be 'crossover_singlesplit', 'crossover_doublesplit', 
                'crossover_uniformsplit' or 'random'
            mutation_rate: float
                Mutation rate
            mutation_type: str
                Mutation type. Values can be 'scramble', 'swap', 'random_reset', 'inversion' or 'random'
            num_generations: int
                Number of generations to train for
            parallel: bool
                Whether to run the genetic algorithm in parallel
            plot: bool
                Whether to plot the metrics
            settings: dict
                Neural network settings. eg layer sizes, activation functions, etc
            description: str
                Description of the model. This data is saved to the ModelDetails.csv file.
            save_frequency: int
                How often to save the model. The model is saved every save_frequency generations.
            random_type: str
                Type of random agent. Values can be 'fixed', 'linear', 'exponential', 'gaussian' or None
            initial_random_rate: float
                Initial random rate. This is the probability of a random agent being selected.
            run_tests: bool
                Whether to run tests
            str_test_folder: str
                Folder to save the test results

        """
        self.env = gym.make(environment)
        self.environment = environment
        self.population_size = population_size
        self.population = self.init_population(environment)
        self.sparse_reward = sparse_reward
        self.fitness_sharing = fitness_sharing
        self.num_select_agents = num_select_agents
        self.selection_type = selection_type
        self.crossover_rate = crossover_rate
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.mutation_method = mutation_method
        self.num_generations = num_generations
        self.description = description
        self.parallel = parallel
        self.plot = plot
        self.settings = settings
        self.terminated = False
        self.save_frequency = save_frequency
        self.random_type = random_type
        self.initial_random_rate = initial_random_rate
        self.run_tests = run_tests
        self.str_test_folder = str_test_folder


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
            self.threshold = 999
            agentPopulation = [NeuralNetwork(2, 3) for _ in range(self.population_size)]
            # raise ValueError('Environment doesn"t quite work yet. Reward is always < 0 and is too random.')
        elif env == 'CartPole-v1':
            self.threshold = 1000
            agentPopulation = [NeuralNetwork(4, 2) for _ in range(self.population_size)]
        elif env == "LunarLander-v2":
            self.threshold = 200
            agentPopulation = [NeuralNetwork(8, 4) for _ in range(self.population_size)]
            

        else:
            raise ValueError('Environment not supported')
        
        return agentPopulation    


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
        self.generation = 1

        # Initialize metrics
        self = initialise_metrics(self)
        self = update_model_details(self)
        
        while self.generation <= self.num_generations:
            print('----------------------------------------------------------------------')
            print('Generation {}'.format(self.generation))

            # Evaluate fitness
            self = evaluate_fitness(self)

            if self.terminated:
                break

            # Selection
            self = selection(self)

            # Crossover
            self = crossover(self)

            # Mutation
            self = mutate(self)

            # Add random agents
            self = random_agent(self)

            # Move to next generation
            self.generation += 1
        
            # Save population
            if self.generation % self.save_frequency == 0:
                save_generation(self)

        save_generation(self)

        # Update model details with final generation
        self = update_model_details(self)

        # Plot metrics
        if self.plot:
            plot_metrics(self)



if __name__ == "__main__":
    
    ga = GeneticAlgorithm(
        environment='CartPole-v1',
        # environment='MountainCar-v0',
        # environment='LunarLander-v2',
        # num_generations=30,
        # population_size=10,
        # save_frequency=5
    )

    # Run genetic algorithm
    ga.run()

