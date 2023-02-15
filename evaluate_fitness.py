import numpy as np
import random
import gym


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
        """
        # Initialize population fitness list.
        population_fitness = []

        # Iterate through the population
        for index, agent in enumerate(self.population):
            print('Status: {}/{}'.format(index+1, len(self.population)))
            
            # Create a new environment and initialize variables
            env = gym.make(self.environment)
            observation = env.reset()
            done = False

            # Initialize fitness score
            fitness = 0

            # Pass agent through environment. Fitness is the sum of rewards. 
            # This section can be tampered with a lot
            while not done:
                # Get action from agent and pass it to the environment
                action = agent.predict_action(observation)
                observation, reward, done, info = env.step(action)

                # Decide what type of fitness function to use here
                ############### This bit can be tampered with a lot ###############
                fitness += reward

            # Print fitness score
            agent.fitness = fitness
            population_fitness.append(fitness)

        # Print the average fitness of the population
        print('Mean fitness: {}'.format(np.mean(population_fitness)))
        print('Best fitness: {}'.format(np.max(population_fitness)))
        print('Best agent index: {}'.format(np.argmax(population_fitness)))

        # Save the population fitness
        self.mean_fitness = int(np.mean(population_fitness))
        self.best_fitness = int(np.max(population_fitness))
        self.best_agent = int(np.argmax(population_fitness))

        # Return the population fitness
        return self, population_fitness