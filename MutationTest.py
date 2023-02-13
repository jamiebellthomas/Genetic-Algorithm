from NeuralNetwork import NeuralNetwork

import random
import numpy as np

import gym

network = NeuralNetwork(2, 3)

#print(network.model.get_weights())
#print(network.model.get_weights()[0].shape)
#print(network.model.get_weights()[1].shape)
#print(network.model.get_weights()[2].shape)
#print(network.model.get_weights()[3].shape)


def flatten(individual):
    """ Mutation """
    # Flatten weights
    print(type(individual))
    weights = individual.model.get_weights()
    weights = [w.flatten() for w in weights]
    weights = np.concatenate(weights)
    print(type(weights))
    return weights

def mutate(flattened_weights, mutation_rate):
    """ Mutation """
    # Mutate weights
    for i in range(len(flattened_weights)):
        flattened_weights[i] *= 1+(random.uniform(-mutation_rate, mutation_rate))
    print(flattened_weights.shape)
    return flattened_weights

#flatten(network)
#mutate(flatten(network), 0.1)

print (network.flatten())