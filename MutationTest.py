from GeneticAlgorithm import NeuralNetwork

import numpy as np

import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
network = NeuralNetwork(2, 3)
print(network.model.get_weights())

def flatten(individual):
    """ Mutation """
    # Flatten weights
    weights = individual.model.get_weights()
    weights = [w.flatten() for w in weights]
    weights = np.concatenate(weights)
    return weights

print (flatten(network))

print (flatten(network).shape)