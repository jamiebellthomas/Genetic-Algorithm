import random
import numpy as np

def crossover(self):
    """ Crossover
    This function performs crossover on the selected population in as many different random way 
    in order to fill the population back up to the original size.
    
    parameters:
    ----------------
        self: GeneticAlgorithm object

    returns:
    ----------------
        offspring: list <np.array>
            Flattened weights of the offspring
        """

    # Number of offspring to create per crossover
    num_offspring = 2

    # create a list with three different crossover methods
    crossover_methods = ['crossover_singlesplit', 'crossover_doublesplit', 'crossover_uniformsplit']

    # choose a random crossover method
    # Implement a crossover rate here by using a random number between 0 and 1 and comparing it to the crossover rate
    if random.random() < self.crossover_rate:
        if self.crossover_method == 'random' or self.crossover_method not in crossover_methods:
            crossover_method = random.choice(crossover_methods)
            print('Crossover method (random): {}'.format(crossover_method))
        else:
            crossover_method = self.crossover_method
            print('Crossover method: {}'.format(crossover_method))
    else:
        print('Crossover method: None')
        crossover_method = None
        
    # This finds the neural network for the parents from the population
    selected_parents = [nn for nn in self.population if nn.selected]

    # Create a list of the nn population minus the selected parents
    offspring_nn = [nn for nn in self.population if not nn.selected]

    # Counter for updated offspring nn
    offspring_counter = 0
    
    # Crosover from the selected popuplation to fill the population back up to the original size
    while offspring_counter < len(offspring_nn):
        # Select the parents for crossover
        parent1 = selected_parents[random.choice(range(len(selected_parents)))]
        parent2 = selected_parents[random.choice(range(len(selected_parents)))]

        if crossover_method == 'crossover_singlesplit':
            # This performs the crossover on the parents with one split
            # For the number of offspring to create per crossover choose a random point to split the parents
            for _ in range(num_offspring):
                # This selects a point at random to split the parents for the weights and biases
                split_weights = random.randint(0,len(parent1.weights)-1)
                split_biases = random.randint(0,len(parent1.biases)-1)

                # Randomly select which parent to take the first half of the weights and biases from
                if random.choice([True, False]):
                    # Takes parent1's first half of weights and biases and parent2's second half of weights and biases
                    child_weights = np.array(parent1.weights[0:split_weights].tolist() + parent2.weights[split_weights:].tolist())
                    child_biases = np.array(parent1.biases[0:split_biases].tolist() + parent2.biases[split_biases:].tolist())
                else:
                    # Takes parent2's first half of weights and biases and parent1's second half of weights and biases
                    child_weights = np.array(parent2.weights[0:split_weights].tolist() + parent1.weights[split_weights:].tolist())
                    child_biases = np.array(parent2.biases[0:split_biases].tolist() + parent1.biases[split_biases:].tolist())

                try:
                    # Edit the offspring nn with the new weights and biases
                    offspring_nn[offspring_counter].weights = child_weights
                    offspring_nn[offspring_counter].biases = child_biases

                    # Increment the offspring counter
                    # offspring_counter += num_offspring
                    offspring_counter += 1
                except:
                    break


        elif crossover_method == 'crossover_doublesplit':
            # This performs the crossover on the parents with two splits
            # For the number of offspring to create per crossover choose two random points to split the parents
            for _ in range(num_offspring):
                # This selects two points at random to split the parents for the weights and biases
                split_weights1 = random.randint(0,len(parent1.weights)-1)
                split_weights2 = random.randint(split_weights1+1,len(parent1.weights))
                split_biases1 = random.randint(0,len(parent1.biases)-1)
                split_biases2 = random.randint(split_biases1+1,len(parent1.biases))

                # Create a random list of choices between parent 0 and 1 for the first, middle and last part
                chosen_parent = [random.choice([0,1]) for _ in range(3)]

                # Make sure the chosen parent is not the same for all parts
                while chosen_parent[0] == chosen_parent[1] == chosen_parent[2]:
                    chosen_parent = [random.choice([0,1]) for _ in range(3)]

                # Use the chosen parent to select the weights and biases for the child
                # First part
                if chosen_parent[0] == 0:
                    parent_first = parent1
                else:
                    parent_first = parent2

                # Middle part
                if chosen_parent[0] == 0:
                    parent_middle = parent1
                else:
                    parent_middle = parent2

                # Last part
                if chosen_parent[0] == 0:
                    parent_last = parent1
                else:
                    parent_last = parent2

                # Create the child weights and biases
                child_weights = np.array(parent_first.weights[0:split_weights1].tolist() + parent_middle.weights[split_weights1:split_weights2].tolist() + parent_last.weights[split_weights2:].tolist())
                child_biases = np.array(parent_first.biases[0:split_biases1].tolist() + parent_middle.biases[split_biases1:split_biases2].tolist() + parent_last.biases[split_biases2:].tolist())

                try:
                    # Edit the offspring nn with the new weights and biases
                    offspring_nn[offspring_counter].weights = child_weights
                    offspring_nn[offspring_counter].biases = child_biases

                    # Increment the offspring counter
                    offspring_counter += 1
                except:
                    break

        elif crossover_method == 'crossover_uniformsplit':
            # This performs the crossover on the parents with uniform splits

            # For the number of offspring to create per crossover run through the weights and biases and randomly select a parent
            for _ in range(num_offspring):
                # Create a list of the weights and biases for the child
                child_weights = []
                child_biases = []

                # Run through the weights and biases and randomly select a parent
                for i in range(len(parent1.weights)):
                    if random.choice([True, False]):
                        child_weights.append(parent1.weights[i])
                    else:
                        child_weights.append(parent2.weights[i])

                for i in range(len(parent1.biases)):
                    if random.choice([True, False]):
                        child_biases.append(parent1.biases[i])
                    else:
                        child_biases.append(parent2.biases[i])

                try:
                    # Edit the offspring nn with the new weights and biases
                    offspring_nn[offspring_counter].weights = np.array(child_weights)
                    offspring_nn[offspring_counter].biases = np.array(child_biases)

                    # Increment the offspring counter
                    offspring_counter += 1
                except:
                    break

        elif crossover_method == None:
            for i in range(num_offspring):
                # pick which parent to copy
                if random.choice([True, False]):
                    parent = parent1
                else:
                    parent = parent2

                try:

                    # Just append a copy of one parent to the offspring
                    offspring_nn[offspring_counter].weights = np.array(parent.weights)
                    offspring_nn[offspring_counter].biases = np.array(parent.biases)

                    offspring_counter += 1
                except:
                    break
         



    # Create the new population by combining the offspring and selected parents lists
    offspring = offspring_nn + selected_parents
    # Check that the offspring is the correct size if it is append it to the population
    if len(offspring) != self.population_size:
        raise ValueError('The offspring is length {} and should be length {}'.format(len(offspring), self.population_size))
    else:
        self.population = offspring

    return self