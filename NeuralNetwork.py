from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input

import numpy as np

class NeuralNetwork():
    """ Neural Network class """
    def __init__(self, input_size, output_size, settings=None):
        """ 
        Constructor. 
        Possible addition: allow architecture to be a dynamic parameter.

        parameters:
        ----------------
            input_size: int
                Number of input nodes
            output_size: int
                Number of output nodes
            settings: dict
                Dictionary of settings for the neural network
        """
        if settings is None:
            layer_size = 5
            num_layers = 1

        # Input layer with input_size nodes, dense layer with 5 nodes and output layer with output_size nodes

        input_layer  = Input(input_size)
        dense_layer1 = Dense(layer_size, activation="relu")
        output_layer = Dense(output_size, activation="linear")
        
        # Assign layers to the model
        model = Sequential()
        model.add(input_layer)

        # Add dense layers
        for _ in range(num_layers):
            model.add(dense_layer1)
        
        model.add(output_layer)

        # Create random weights
        # Sets random weights to the model's weights and biases
        weights = model.get_weights()
        weights = [np.random.rand(*w.shape) for w in weights]
        model.set_weights(weights)

        # Assign model to class
        self.model = model
        self.layers = model.layers

        # Store hyperparameters
        self.num_layers = num_layers
        self.layer_size = layer_size

        # Assign the weights and biases to the class for easy access
        self.weights = {}
        self.biases = {}
        for i, layer in enumerate(self.layers):
            self.weights[i] = layer.get_weights()[0]
            self.biases[i] = layer.get_weights()[1]


    # Function that takes the observation of the state as input and returns the action
    def predict_action(self, observation):
        """ Predict
        This function returns the action based on the observation.

        parameters:
        ----------------
            observation: np.array
                Observation array

        returns:
        ----------------
            action: int
                Action to be taken
        """
        # Choose the action with the highest value
        action = np.argmax(self.model.predict(observation.reshape(1, -1)))

        return action