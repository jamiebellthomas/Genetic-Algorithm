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

def dynamic_mutation_probability(self):
    """
    Dynamic mutation probability
    For selected agents, the fitness is extracted and normalised. The normalised fitness is then used to calculate the
    mutation probability for each agent.
    """
    max_selected_fitness = -np.inf
    # Find the maximum fitness of the selected agents
    for agent in self.population:
        if agent.selected:
            if agent.fitness > max_selected_fitness:
                max_selected_fitness = agent.fitness
    for agent in self.population:
        if agent.selected:
            # Calculate the normalised fitness
            normalised_fitness = agent.fitness/max_selected_fitness
            # Calculate the mutation probability
            agent.selected_mutation_rate = 1 - normalised_fitness

    return self
            
    
    return

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
    # Mutation
    
    if random.uniform(0,1) < mutation_rate:
        # Randomly select a mutation method and apply it
        mutation_method = random.randint(0,3)
        if mutation_method == 0:
            flattened_weights = scramble_mutation(flattened_weights, mutation_rate)
        elif mutation_method == 1:
            flattened_weights = swap_mutation(flattened_weights, mutation_rate)
        elif mutation_method == 2:
            flattened_weights = random_reset_mutation(flattened_weights, mutation_rate)
        elif mutation_method == 3:
            flattened_weights = inversion_mutation(flattened_weights, mutation_rate)

    return flattened_weights
        
        
    

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


def mutate(self):
    """
    This will mutate the weights of the neural network. The mutation rate is determined by whether the agent weights are selected
    from the prevoius generation of created by crossover.

    Those created by crossover will have a fixed mutation rate defined in the GeneticAlgorithm initialisation, 
    those selected from the previous generation will have a dynamic mutation rate determined by the fitness of the agent.
    """
    for agent in self.population:
        if agent.selected:
            agent.weights = mutate_gene(flattened_weights=agent.weights, mutation_rate=agent.selected_mutation_rate)
            agent.biases = mutate_gene(flattened_weights=agent.biases, mutation_rate=agent.selected_mutation_rate)
        else:
            agent.weights = mutate_gene(flattened_weights=agent.weights, mutation_rate=self.mutation_rate)
            agent.biases = mutate_gene(flattened_weights=agent.biases, mutation_rate=self.mutation_rate)
    agent.update_weights_biases()
    return self

def scramble_mutation(chromosome, mutation_rate):
    """
    Scramble mutation
    Takes a block of genes and randomly reorders them
    """
    # Take out a list of genes from the chromosome, scramble it, and put it back in
    gene_indices_to_scramble = []
    for i in range(len(chromosome)):
        if random.uniform(0,1) < mutation_rate:
            gene_indices_to_scramble.append(i)
    scrambled_indices = random.shuffle(gene_indices_to_scramble)
    for i in range(len(gene_indices_to_scramble)):
        chromosome[gene_indices_to_scramble[i]] = chromosome[scrambled_indices[i]]
    return chromosome

def swap_mutation(chromosome, mutation_rate):
    """
    Swap mutation
    Takes two genes and swaps them
    """
    for i in range(len(chromosome)):
        if random.uniform(0,1) < mutation_rate:
            index1 = i
            index2 = index1
            while index2 == index1 or index2 == i:
                index2 = random.randint(0,(len(chromosome)-1))
            chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]
    return chromosome

def inversion_mutation(chromosome, mutation_rate):
    """
    Inversion mutation
    Takes a continuous block of genes and reverses their order
    """
    for i in range(len(chromosome)):
        if random.uniform(0,1) < mutation_rate:
            start_index = i
            end_index = i + random.randint(1,5)
            chromosome[start_index:end_index] = chromosome[start_index:end_index][::-1]

    return chromosome

def random_reset_mutation(chromosome, mutation_rate):
    """
    Random reset mutation
    Takes a gene and randomly resets it to a new value within the range of the chromosome
    """
    for i in range(len(chromosome)):
        if random.uniform(0,1) < mutation_rate:
            chromosome[i] = random.uniform(min(chromosome), max(chromosome))
    return chromosome

def selection_mutation(self):
    """
    Selection mutation
    Takes the best agent and randomly mutates it
    """
    return