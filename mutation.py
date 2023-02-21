from NeuralNetwork import NeuralNetwork

import random
import numpy as np


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
    

def mutate_gene(flattened_weights, mutation_rate, mutation_method):
    ''' 
    Mutation
    
        parameters:
        ----------------
            flattened_weights: 1D array of weights
            mutation rate: float
            mutation method: string
        ----------------
        returns:
            mutated weights: 1D array of weights
    '''
    # Mutation
    # If you want to make the mutations random on each agent in each generation then uncomment the following line
    
    # mutation_methods = ['scramble', 'swap', 'random_reset', 'inversion']
    # mutation_method = random.choice(mutation_methods)
    
    if random.uniform(0,1) < mutation_rate:
        # Choose a mutation method
        if mutation_method == 'scramble':
            flattened_weights = scramble_mutation(flattened_weights, mutation_rate)
        elif mutation_method == 'swap':
            flattened_weights = swap_mutation(flattened_weights, mutation_rate)
        elif mutation_method == 'random_reset':
            flattened_weights = random_reset_mutation(flattened_weights, mutation_rate)
        elif mutation_method == 'inversion':
            flattened_weights = inversion_mutation(flattened_weights, mutation_rate)

    return flattened_weights


def mutate(self):
    """
    This will mutate the weights of the neural network. The mutation rate is determined by whether the agent weights are selected
    from the prevoius generation of created by crossover.

    Those created by crossover will have a fixed mutation rate defined in the GeneticAlgorithm initialisation, 
    those selected from the previous generation will have a dynamic mutation rate determined by the fitness of the agent.
    """
    mutation_methods = ['scramble', 'swap', 'random_reset', 'inversion']

    if self.mutation_method == 'random' or self.mutation_method not in mutation_methods:
        mutation_method = random.choice(mutation_methods)
        print('Mutation method (random): {}'.format(mutation_method))
    else:
        mutation_method = self.mutation_method
        print('Mutation method: {}'.format(mutation_method))
    
    for agent in self.population:
        if agent.selected:
            agent.weights = mutate_gene(flattened_weights=agent.weights, mutation_rate=agent.selected_mutation_rate, mutation_method=mutation_method)
            agent.biases = mutate_gene(flattened_weights=agent.biases, mutation_rate=agent.selected_mutation_rate, mutation_method=mutation_method)
        else:
            agent.weights = mutate_gene(flattened_weights=agent.weights, mutation_rate=self.mutation_rate, mutation_method=mutation_method)
            agent.biases = mutate_gene(flattened_weights=agent.biases, mutation_rate=self.mutation_rate, mutation_method=mutation_method)

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
    if len(gene_indices_to_scramble) == 0:
        return chromosome
    scrambled_indices = random.sample(gene_indices_to_scramble, len(gene_indices_to_scramble))
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