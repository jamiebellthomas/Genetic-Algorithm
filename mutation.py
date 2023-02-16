from NeuralNetwork import NeuralNetwork

import random
import numpy as np




def flatten(network):
    ''' 
    Flatten weights
    
        parameters:
        ----------------
            network: NeuralNetwork object to be flattened
        ----------------
        returns:
            weights vector: 1D array of weights
    '''
    # Flatten weights
    weights = network.model.get_weights()
    weights = [w.flatten() for w in weights]
    weights = np.concatenate(weights)
    return weights


def mutate_gene(flattened_weights, mutation_rate):
    ''' 
    Mutation
    
        parameters:
        ----------------
            flattened_weights: 1D array of weights
            mutation rate: float
        ----------------
        returns:
            mutated weights: 1D array of weights
    '''
    mutated = [0] * len(flattened_weights) 
    # Mutate weights
    for i,v in enumerate(flattened_weights):
        mutated[i] = v * (random.uniform(-mutation_rate, mutation_rate) + 1)
    return mutated
    

def unflatten(flattened_weights, network):
    ''' 
    Unflatten weights
    
        parameters:
        ----------------
            flattened_weights: 1D array of weights
            network: NeuralNetwork object to be written over
        ----------------
        returns:
            network: NeuralNetwork object with unflattened weights
    '''
    # Potential error cathing idea
    #if sum([layer.size for layer in network.model.get_weights()]) != flattened_weights.size:
    #    print('Error: flattened weights do not match network architecture')
    #    return
    appended_weights = 0
    weight_lists = []
    for i in range(len(network.model.get_weights())):
        layer_size = network.model.get_weights()[i].size
        layer_dimensions = network.model.get_weights()[i].shape
        weight_lists.append(np.array(flattened_weights[appended_weights:appended_weights+layer_size]).reshape(layer_dimensions))
        appended_weights += layer_size
    network.model.set_weights(weight_lists)
    return network


def mutate(self,flattened_weights:list):
    """ 
    This function mutates the weights of a given agent by a random amount between -mutation_rate and mutation_rate

    parameters: 
        list of column vectors of weights for each agent
    returns: 
        mutated neural networks for each agent
    """
    next_gen = []

    for i in range(self.population_size):   
        mutated_vector = mutate_gene(flattened_weights[i], self.mutation_rate)
        new_network = unflatten(mutated_vector, self.population[i])  
        next_gen.append(new_network)
    
    return next_gen