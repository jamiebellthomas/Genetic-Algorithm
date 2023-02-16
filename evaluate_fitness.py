import numpy as np
import random
import gym
import multiprocessing as mp
import time
from functools import partial

def evaluate_agent(self, input):
    index, agent = input

    # Print the agent index out of the population size
    print('-------------------------Evaluating agent {}/{}-------------------------'.format(index+1, self.population_size))

    # Create a new environment and initialize variables
    env = gym.make(self.environment)
    observation = env.reset()
    done = False

    # Convert observation to a 2D array if it is a tuple
    if type(observation) == tuple:
        observation = observation[0]

    # Initialize fitness score
    fitness = 0

    # Pass agent through environment. Fitness is the sum of rewards. 
    # This section can be tampered with a lot
    while not done:
        # Get action from agent and pass it to the environment
        action = agent.predict_action(observation)

        # Try with 4 outputs if it errors except with 5 outputs this will depend on version of gym
        try:
            observation, reward, done, info = env.step(action)
        except:
            observation, reward, done, truncate, info = env.step(action)

        # Decide what type of fitness function to use here
        ############### This bit can be tampered with a lot ###############
        fitness += reward

    return fitness

from metrics import update_metrics

def evaluate_fitness(self):
        """ 
        This function evaluates the fitness of the population. The fitness is currently the sum of rewards.
        Each agent is loaded and passed through the environment 
        
        parameters:
        ----------------
            self: GeneticAlgorithm
                GeneticAlgorithm object
        returns:
        ----------------
            self: GeneticAlgorithm
                Updated GeneticAlgorithm object
            population_fitness: list <float>
                List of fitness scores for each agent in the population
            terminated: bool
                Flag to indicate if the threshold has been met
        """
        start = time.time()
        pool_input = list(enumerate(self.population))

        # Evaluate the fitness of each agent in the population in parallel or serial
        if self.parallel:
            cores = mp.cpu_count() -1
            print('Evaluating fitness in parallel with {} cores'.format(cores))
            pool_obj = mp.Pool()

            # Evaluate the fitness of each agent in the population
            population_fitness = pool_obj.map(partial(evaluate_agent, self), pool_input)

            pool_obj.close()
        else:
            population_fitness = []
            for input in pool_input:
                fitness = evaluate_agent(self, input)
                population_fitness.append(fitness)
        
        print('Population fitness: {}'.format(population_fitness))

        # Print the average fitness of the population
        print('Mean fitness: {}'.format(np.mean(population_fitness)))
        print('Best fitness: {}'.format(np.max(population_fitness)))
        print('Best agent index: {}'.format(np.argmax(population_fitness)))

        # Save the population fitness
        self.mean_fitness = int(np.mean(population_fitness))
        self.best_fitness = int(np.max(population_fitness))
        self.best_agent = int(np.argmax(population_fitness))
        
        end = time.time()
        self.duration = end - start

        self.all_fitness = population_fitness

        # Update the metrics
        self = update_metrics(self)

        # If threshold is met, pass terminated flag
        if self.best_fitness >= self.threshold:
            terminated = True
        else:
            terminated = False

        # Return the population fitness
        return self, population_fitness, terminated