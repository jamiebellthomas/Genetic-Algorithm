from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input

import numpy as np

class NeuralNetwork():
    """ Neural Network class """
    def __init__(self, input_size, output_size):
        """ 
        Constructor. 
        Possible addition: allow architecture to be a dynamic parameter.

        parameters:
        ----------------
            input_size: int
                Number of input nodes
            output_size: int
                Number of output nodes
        """
        # Input layer with input_size nodes, dense layer with 5 nodes and output layer with output_size nodes

        input_layer  = Input(input_size)
        dense_layer1 = Dense(5, activation="relu")
        output_layer = Dense(output_size, activation="linear")
        
        # Assign layers to the model
        model = Sequential()
        model.add(input_layer)
        model.add(dense_layer1)
        model.add(output_layer)

        # Create random weights
        # Sets random weights to the model's weights and biases
        weights = model.get_weights()
        weights = [np.random.rand(*w.shape) for w in weights]
        model.set_weights(weights)

        self.model = model
        self.layers = model.layers