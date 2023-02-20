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
            observation, reward, done, _, info = env.step(action)

        # Decide what type of fitness function to use here
        ############### This bit can be tampered with a lot ###############
        fitness += reward


        ############### Introduce fitness for sparse rewards ###############
        '''
        This section is for sparse rewards. It is not currently used but can be used if the environment
        is similar to the LunarLander-v2 environment or MountainCar-v0 environment.

        The idea is to give a reward for each new state that the agent visits. This is done by keeping
        track of the states that the agent has visited.

        parameters:
        ----------------
            self: GeneticAlgorithm
                GeneticAlgorithm object
            observation: list <float>
                List of observations from the environment
        returns:
        ----------------
            fitness: float
                Fitness score for the agent

        '''
        # define a frontier function to keep track of the states visited
        '''
        the frontier is a n dimensional volume of space that the agent has visited.
        it will be updated every time the agent visits a new state.

        frontier will have a length of n where n is the number of dimensions in the state space
        in form:
        frontier = [ 
                    [x1_min, x1_max], # dimension 1
                    [x2_min, x2_max] # dimension 2
                                                 ...]
        
        '''
        if self.sparse_reward == True:

            num_dimensions = len(observation)
            frontier = []

            # initialise the frontier
            for i in range(num_dimensions):
                    frontier.append([observation[i], observation[i]])

            '''
            The actual reward function needs to be updated. Currently it is just giving a reward for
            visiting a new state. It should be updated to give a reward for visiting a new state that
            is not too far away from the current state. This will encourage the agent to explore
            the state space more.
            '''
            # update the frontier
            for i in range(num_dimensions):
                if observation[i] < frontier[i][0]:

                    '''
                    basic case:
                    fitness += 1
                    
                    But in this case:
                    the reward is proportional to the distance from the current state to the frontier
                    this will encourage the agent to explore the state space more

                    '''
                    fitness += abs(frontier[i][0] - observation[i]) / frontier[i][0]

                    frontier[i][0] = observation[i]
                elif observation[i] > frontier[i][1]:

                    # give reward for visiting new state
                    fitness += abs(observation[i] - frontier[i][1]) / frontier[i][1]
                    
                    frontier[i][1] = observation[i]



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
            # Create a pool of processes
            cores = mp.cpu_count() - 1
            print('Evaluating fitness in parallel with {} cores'.format(cores))
            pool_obj = mp.Pool(processes=cores)          

            # Evaluate the fitness of each agent in the population
            population_fitness = pool_obj.map(partial(evaluate_agent, self), pool_input)

            pool_obj.close()
        else:
            population_fitness = []
            for input in pool_input:
                fitness = evaluate_agent(self, input)
                population_fitness.append(fitness)

        ######## FITNESS SHARING ########
        '''
        Fitness sharing is a method of preventing premature convergence in genetic algorithms.
        It is a method of penalizing agents that are too similar to each other. This is done by
        reducing the fitness of agents that are too similar to each other.

        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.9077&rep=rep1&type=pdf

        The fitness sharing method used here is based on the euclidean distance between agents
        and will require flattening the agents weights: 

        ############################################################
        TO BE DISCUSSED move the flattening to the evaluate_fitness.py
        ############################################################

        parameters:
        ----------------
            self: GeneticAlgorithm
                GeneticAlgorithm object
        returns:
        ----------------
            population_fitness: list <float>
                List of fitness scores for each agent in the population


        '''
        if self.fitness_sharing == True:
            # import the flatten function
            from mutation import flatten

            # define fitness sharing parameters - these can be changed
            shar_param = 0.5 # a parameter that controls the strength of the fitness sharing
            shar_range = 5 # a euclidean distance between agents

            new_fitness = []
            print('Applying fitness sharing')

            # for each agent in the population
            for i, agent in enumerate(self.population):
                # for each other agent in the population
                denominator = 1
                chosen_agent = flatten(agent)

                for j, other in enumerate(self.population):
                    # if the agent is not the same as the other agent
                    if i != j:
                        # retrieve the weights from the agent
                        other_agent = flatten(other)

                        # calculate the euclidean distance between the agents 
                        # NOTE: this can be changed to the manhattan distance or maximum distance
                        # ord = 2 is euclidean , ord = 1 is manhattan, ord = np.inf is max distance

                        distance = np.linalg.norm(chosen_agent - other_agent, ord=2)

                        # if the distance is less than the shar_range
                        if distance < shar_range:
                            # calculate the fitness sharing term
                            denominator += (1 - distance/shar_range)**(shar_param)

                # calculate the new fitness
                new_fitness.append(population_fitness[i]/denominator)
                
            # replace the population fitness with the new fitness
            population_fitness = new_fitness
            print('Fitness sharing applied')



    
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