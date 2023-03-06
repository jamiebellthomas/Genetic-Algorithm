import numpy as np
def random_probability(self):
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


def randomise_weights(network):
    
        

    pass

def random_agent(self):
    for agent in self.population:
        if agent.selected == False:
            pass
        agent.update_weights_biases()
    pass