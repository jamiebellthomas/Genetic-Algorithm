import numpy as np
import random
import gym
import multiprocessing as mp
import time
from functools import partial
from metrics import update_metrics


def sparse_rewards(observation, fitness, frontier):
    '''
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

    the frontier is a n dimensional volume of space that the agent has visited.
    it will be updated every time the agent visits a new state.

    frontier will have a length of n where n is the number of dimensions in the state space
    in form:
    frontier = [ 
                [x1_min, x1_max], # dimension 1
                [x2_min, x2_max] # dimension 2
                                                ...]

    rewards will be greater for states that are farther away from the frontier
    
    '''
    num_dimensions = len(observation)

    for i in range(num_dimensions):
        # add random noise to reward to prevent premature convergence
        noise = random.uniform(-0.05, 0.05)

        if observation[i] < frontier[i][0] and frontier[i][0] != 0:

            # looking for the distance between the observation and the starting point 
            # to make the reward greater for states that are farther away from the frontier

            fitness += (abs((frontier[i][0] - observation[i])/ frontier[i][0]))**0.5 + noise

            # frontier[i][0] = observation[i]

        elif observation[i] > frontier[i][1] and frontier[i][1] != 0:

            fitness += (abs((observation[i] - frontier[i][1])/ frontier[i][1]))**0.5 + noise

            # frontier[i][1] = observation[i]

    return frontier, fitness


def evaluate_agent(self, input):
    index, agent = input

    # Print the agent index out of the population size
    print('-------------------------Evaluating agent {}/{}-------------------------'.format(index+1, self.population_size))

    # Create a new environment and initialize variables
    env = self.env
    observation = env.reset()
    done = False
    truncation = False

    # Convert observation to a 2D array if it is a tuple
    if type(observation) == tuple:
        observation = observation[0]

    # Initialize fitness score and frontier
    fitness = 0
    frontier = []

    for i in range(len(observation)):
        frontier.append([observation[i], observation[i]]) # [min, max]

    # Pass agent through environment. Fitness is the sum of rewards. 
    iter = 0
    while not done:
        # Get action from agent and pass it to the environment
        action = agent.predict_action(observation)

        # Try with 4 outputs if it errors except with 5 outputs this will depend on version of gym
        try:
            observation, reward, done, info = env.step(action)
        except:
            observation, reward, done, truncation, info = env.step(action)


        # Decide what type of fitness function to use here
        fitness += reward

        if done or truncation:
            break

        #if iter >= 100 and fitness < 0 and not self.env == 'LunarLander-v2' :
        #    self.sparse_reward = True 
 
        # If sparse rewards are enabled then update the frontier
        if self.sparse_reward:
            frontier, fitness = sparse_rewards(observation, fitness, frontier)

        # Print progress
        iter += 1
        if iter % 100 == 0:
            print('Iteration: {} | Fitness: {}'.format(iter, fitness))

    # Update the agent fitness
    agent.fitness = fitness

    return self


def fitness_sharing(self):
    ######## FITNESS SHARING ########
    '''
    Fitness sharing is a method of preventing premature convergence in genetic algorithms.
    It is a method of penalizing agents that are too similar to each other. This is done by
    reducing the fitness of agents that are too similar to each other.

    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.9077&rep=rep1&type=pdf

    The fitness sharing method used here is based on the euclidean distance between agents
    and will require flattening the agents weights: 

    parameters:
    ----------------
        self: GeneticAlgorithm
            GeneticAlgorithm object
    returns:
    ----------------
        population_fitness: list <float>
            List of fitness scores for each agent in the population


    '''
    # define fitness sharing parameters - these can be changed
    shar_param = 0.5 # a parameter that controls the strength of the fitness sharing
    shar_range = 10 # a euclidean distance between agents

    print('Applying fitness sharing')

    # for each agent in the population
    for i, agent in enumerate(self.population):
        
        denominator = 1

        # retrieve the flattened weights from the agent
        chosen_agent = agent.weightsnbiases

        # calculate shar_range as 10% of the length of the agent
        shar_range = len(chosen_agent)*0.1

        # for each other agent in the population
        for j, other in enumerate(self.population):
            # if the agent is not the same as the other agent
            if i != j:
                # retrieve the weights from the agent
                other_agent = other.weightsnbiases

                # calculate the manhattan distance between the agents 
                # NOTE: this can be changed to the manhattan distance or maximum distance
                # ord = 2 is euclidean , ord = 1 is manhattan, ord = np.inf is max distance

                distance = np.linalg.norm(chosen_agent - other_agent, ord=1)

                # if the distance is less than the shar_range
                if distance < shar_range:
                    # calculate the fitness sharing term
                    denominator += (1 - distance/shar_range)**(shar_param)

        # calculate the new fitness
        # Add the new fitness to the agent
        agent.fitness = agent.fitness/denominator
        
    # replace the population fitness with the new fitness
    print('Fitness sharing applied')

    return self





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
            pool_obj.map(partial(evaluate_agent, self), pool_input)

            pool_obj.close()
        else:
            for input in pool_input:
                self = evaluate_agent(self, input)
        
        # apply fitness sharing
        if self.fitness_sharing == True:
            self = fitness_sharing(self)

        # Get the fitness of each agent in the population
        population_fitness = [agent.fitness for agent in self.population]
    
        print('Population fitness and indices: {}'.format( list(enumerate(population_fitness))))

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
            self.terminated = True

        # Return the population fitness
        return self


