import numpy as np
import random
def random_probability(self):
    """
    Random probability
    Computes the random probability of a random agent manifesting in the population as a 
    function of the generation number and total number of generations.
    """
    # 4 types of decay
    # 1. Fixed - no decay
    if self.random_type == 'none':
        random_rate = 0
    if self.random_type == 'fixed':
        random_rate = self.initial_random_rate
    # 2. Linear - linear decay
    elif self.random_type == 'linear':
        random_rate = (1-((self.generation-1)/self.num_generations)) * self.initial_random_rate
    # 3. Exponential - exponential decay
    elif self.random_type == 'exponential':
        random_rate = self.initial_random_rate * (1-(self.generation/self.num_generations))*np.exp(-self.generation/self.num_generations)
    # 4. Gaussian - gaussian decay
    elif self.random_type == 'gaussian':
        random_rate = self.initial_random_rate * np.exp(-(self.generation**2/100))
    else:
        print(self.random_type)
        raise ValueError('Invalid random type')
    # These decays represent the decaying probability of a random agent being replacing an unselected agent in the population
    # Which was created during crossover and mutation
    return random_rate

def randomise_weights(flattened_weights):
    """
    Randomise weights in a 1D array of weights
    """
    # Define range of current weights
    max_weight = max(flattened_weights)
    min_weight = min(flattened_weights)
    # Randomise weights within the same range
    for i in range(len(flattened_weights)):
        flattened_weights[i] = np.random.uniform(min_weight, max_weight)
    return flattened_weights    

def random_agent(self):
    """
    Random agent
    Cycles through the population and randomly selects agents to be replaced with a new random agent.
    Probability of reset occuring definied by random_probability()
    """
    # Randomise weights and biases of unselected agents
    # The probability of this occuring is defined by the random_probability() function
    # Which is a function of the generation number and the total number of generations
    for agent in self.population:
        if agent.selected == False:
            random_rate = random_probability(self)
            if random.uniform(0,1) < random_rate:
                agent.weights = randomise_weights(agent.weights)
                agent.biases = randomise_weights(agent.biases)
                agent.update_weights_biases()
                print('Random agent added')
    return self