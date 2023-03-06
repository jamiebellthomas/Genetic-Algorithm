import numpy as np
import random
def random_probability(self):
    """
    Random probability
    Computes the random probability for each agent in the population as a function of the generation number.
    """
    if self.random_type == 'fixed':
        random_rate = self.initial_random_rate
    elif self.random_type == 'linear':
        (1-((self.generation-1)/self.num_generations)) * self.initial_random_rate
    elif self.random_type == 'exponential':
        random_rate = self.initial_random_rate * (1-(self.generation/self.num_generations))*np.exp(-self.generation/self.num_generations)
    elif self.random_type == 'gaussian':
        random_rate = self.initial_random_rate * np.exp(-(self.generation**2/100))
    else:
        raise ValueError('Invalid random type')
    return random_rate


def randomise_weights(flattened_weights):
    """
    Randomise weights in a 1D array of weights
    """
    max_weight = max(flattened_weights)
    min_weight = min(flattened_weights)
    for i in range(len(flattened_weights)):
        flattened_weights[i] = np.random.uniform(min_weight, max_weight)
    return flattened_weights    

def random_agent(self):
    for agent in self.population:
        if agent.selected == False:
            random_rate = random_probability(self)
            if random.uniform(0,1) < random_rate:
                agent.weights = randomise_weights(agent.weights)
                agent.biases = randomise_weights(agent.biases)
                agent.update_weights_biases()
    return self