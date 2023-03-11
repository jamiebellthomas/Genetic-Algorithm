from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input

import numpy as np

class NeuralNetwork():
    """ Neural Network class """
    def __init__(self, input_size, output_size, model=None, settings=None):
        """ 
        Constructor. 
        Possible addition: allow architecture to be a dynamic parameter.

        parameters:
        ----------------
            input_size: int
                Number of input nodes
            output_size: int
                Number of output nodes
            model: tf.keras.Sequential
                When loading model from file, pass the model to the constructor
        """
        if settings is None:
            layer_size = 8
            num_layers = 3
            dense_activation = 'relu'
            output_activation = 'linear'
            settings = {'layer_size': layer_size, 'num_layers': num_layers, 'dense_activation': dense_activation, 'output_activation': output_activation}
        else:
            layer_size = settings['layer_size']
            num_layers = settings['num_layers']
            dense_activation = settings['dense_activation']
            output_activation = settings['output_activation']

        
        # Input layer with input_size nodes, dense layer with 5 nodes and output layer with output_size nodes

        input_layer  = Input(input_size)
        # Adding multiple dense layers in this format didnt work (only added one and stopped.)
        dense_layer = Dense(layer_size, activation=dense_activation)
        output_layer = Dense(output_size, activation=output_activation)
        
        # If model is passed to the constructor, use that model else
        if model is None:
            # Assign layers to the model
            model = Sequential()
            model.add(input_layer)
            # Add dense layers
            for _ in range(num_layers):         
                model.add(Dense(layer_size, activation=dense_activation))
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
        self.get_flattened_weights_biases()     

        # Boolean to check if the network is selected for crossover
        self.selected = False

        # Fitness of the network
        self.fitness = 0

        # Store mutation rate for selected networks
        self.selected_mutation_rate = 0


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

    def get_flattened_weights_biases(self):
        """ Get flattened weights
        This function returns the weights and biases of the neural network as a 1D array.

        returns:
        ----------------
            weights: np.array
                1D array of weights and biases
        """
        # Get the weights and biases
        weights = []
        biases = []

        for layer in self.layers:
            weights.append(layer.get_weights()[0])
            biases.append(layer.get_weights()[1])
        
        # Flatten weights and biases
        weights = [w.flatten() for w in weights]
        self.weights = np.concatenate(weights)

        biases = [b.flatten() for b in biases]
        self.biases = np.concatenate(biases)

        # Concatenate weights and biases
        self.weightsnbiases = np.concatenate((self.weights, self.biases))

        
    def update_weights_biases(self):
        """ Update weights and biases
        This function updates the weights and biases of the neural network using the weights and biases stored in the class.
        It also resets the selected boolean and fitness.
        """
        all_weights = []
        weight_index = 0
        bias_index = 0

        # Loop though the layers and update the weights and biases
        for layer in self.layers:
            # Get the layer sizes and dimensions
            layer_sizes = layer.get_weights()[0].size
            layer_dimensions = layer.get_weights()[0].shape

            # Set the weights and biases for the layer
            all_weights.append(self.weights[weight_index:weight_index+layer_sizes].reshape(layer_dimensions))
            all_weights.append(self.biases[bias_index:bias_index+layer_dimensions[1]].reshape(layer_dimensions[1],))

            # Update the weight and bias indices
            weight_index += layer_sizes
            bias_index += layer_dimensions[1]

        # Set the weights and biases
        self.model.set_weights(all_weights)

        # Reset the selected boolean
        self.selected = False

        # Reset the fitness
        self.fitness = 0

        # Reset the mutation rate
        self.selected_mutation_rate = 0