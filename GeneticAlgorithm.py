import numpy as np
import random
import gym

from selection import selection
from NeuralNetwork import NeuralNetwork
from saving_data import save_generation
from evaluate_fitness import evaluate_fitness
from mutation import mutate, flatten
from metrics import initialise_metrics, update_metrics
from crossover import crossover
from plotting_data import plot_metrics


class GeneticAlgorithm():
    """ Genetic Algorithm class """
    def __init__(self, environment, population_size=5, sparse_reward=False, fitness_sharing=False, selection_type='elitism', crossover_rate=0.7, 
                mutation_rate=0.1, num_generations=5, parallel=False, plot=False, description=None):
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
            selection_type: str
                Selection type. Values can be 'tournament', 'proportional-roulette-wheel', 
                'rank-based-rolette-wheel', 'elitism'
            crossover_rate: float
                Crossover rate
            mutation_rate: float
                Mutation rate
            num_generations: int
                Number of generations to train for
            parallel: bool
                Whether to run the genetic algorithm in parallel
            plot: bool
                Whether to plot the metrics
            description: str
                Description of the model. This data is saved to the ModelDetails.csv file.
        """
        self.environment = environment
        self.population_size = population_size
        self.population = self.init_population(environment)
        self.sparse_reward = sparse_reward
        self.fitness_sharing = fitness_sharing
        self.selection_type = selection_type
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.description = description
        self.parallel = parallel
        self.plot = plot


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
            self.threshold = 500
            agentPopulation = [NeuralNetwork(4, 2) for _ in range(self.population_size)]
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
        
        while self.generation <= self.num_generations:
            print('Generation {}'.format(self.generation))

            # Evaluate fitness
            self, population_fitness, terminated = evaluate_fitness(self)

            if terminated:
                break

            # Selection
            selected_population = selection(self, population_fitness, num_agents=2)

            # Crossover
            offspring = crossover(self, selected_population)

            print(len(offspring))
            # Mutation
            mutated_offspring = mutate(self, offspring)

            # Update population and generation
            self.population = mutated_offspring

            self.generation += 1

        if not terminated:
            # Final evaluation
            evaluate_fitness(self)
        
        # Save data
        save_generation(self)

        # Plot metrics
        if self.plot:
            plot_metrics(self)



if __name__ == "__main__":
    

    ga = GeneticAlgorithm(
        environment='CartPole-v1',
    )

    # Run genetic algorithm
    ga.run()

