import numpy as np
import random
import gym


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