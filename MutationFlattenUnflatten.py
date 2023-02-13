from NeuralNetwork import NeuralNetwork

import random
import numpy as np

import gym
mutation_rate = 0.2
network = NeuralNetwork(2, 3)
#print(network.model.get_weights())
#for i in range(len(network.model.get_weights())):
#    print(network.model.get_weights()[i].shape, f'layer {i+1} shape')
#    print(network.model.get_weights()[i].size, f'layer {i+1} size')
#print(sum([layer.size for layer in network.model.get_weights()]))
#print(network.model.get_weights())
#print(network.model.get_weights()[0].shape)
#print(network.model.get_weights()[1].shape)
#print(network.model.get_weights()[2].shape)
#print(network.model.get_weights()[3].shape)


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

def mutate(flattened_weights, mutation_rate):
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


"""
original_flattened = flatten(network)
original_unflattened = unflatten(original_flattened, network)



print('Original:', network.model.get_weights())
print('Original Remade:', original_unflattened.model.get_weights())



mutated = mutate(original_flattened, 0.1)
mutated_unflattened = unflatten(mutated, network)


print('Original:', original_unflattened.model.get_weights())
print('Mutated:', mutated_unflattened.model.get_weights())
print('Delta:', mutated - original_flattened)
"""