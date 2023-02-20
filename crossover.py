import random
import numpy as np
from mutation import flatten

import random
import numpy as np
from mutation import flatten

def crossover(self, selected_population):
        """ Crossover
        This function performs crossover on the selected population in as many different random way in order to fill the population back up to the original size.
        
        parameters:
        ----------------
            selected_population: list <int>
                List of indices of the selected population

        returns:
        ----------------
            offspring: list <np.array>
                Flattened weights of the offspring
         """
        offspring = []

        # create a list with three different crossover methods
        crossover_methods = ['crossover_singlesplit', 'crossover_doublesplit', 'crossover_uniformsplit']
        
        # Crosover from the selected popuplation to fill the population back up to the original size
        for i in range((self.population_size)//2):
            

            # This selects the integers for the indexing of two parents at random from the selected population
            parent1 = selected_population[random.choice(range(len(selected_population)))]
            parent2 = selected_population[random.choice(range(len(selected_population)))]
           # This checks if the parents are the same and if so, selects a new parent
            if parent1 == parent2:
                parent2 = selected_population[random.choice(range(len(selected_population)))]

            # This finds the neural network for the parents from the population
            parent1 = self.population[parent1]
            parent2 = self.population[parent2]

            # This flattens the weights of the parents from a nn to a np array
            parent1 = flatten(parent1)
            parent2 = flatten(parent2)
          
            #randomly select a crossover method
            crossover_method = random.choice(crossover_methods)
            

            if crossover_method == 'crossover_singlesplit':
                # This performs the crossover on the parents
                
                # This selects a point at random to split the parents
                split = random.randint(0,len(parent1)-1)
                # This creates the children by selecting the first half(up to splitting point) of the first parent and 
                # the second half of the second parent and then inversely for the second child
                child1_genes = np.array(parent1[0:split].tolist() + parent2[split:].tolist())
                child2_genes = np.array(parent2[0:split].tolist() + parent1[split:].tolist())
            
                # append the children to the offspring list
                offspring.append(child1_genes)
                offspring.append(child2_genes)
            
            elif crossover_method == 'crossover_doublesplit':
                # This performs the crossover on the parents
                
                # This selects two points at random to split the parents
                split1 = random.randint(0,len(parent1)-1)
                split2 = random.randint(split1+1,len(parent1))
                
                # This creates the children by selecting the first half(up to splitting point) of the first parent and 
                # the second half of the second parent and then inversely for the second child
                child1_genes = np.array(parent1[0:split1].tolist() + parent2[split1:split2].tolist() + parent1[split2:].tolist())
                child2_genes = np.array(parent2[0:split1].tolist() + parent1[split1:split2].tolist() + parent2[split2:].tolist())
                
                # # print the difference between parent 1 and child 1
                # print('difference between parent 1 and child 1', np.subtract(parent1, child1_genes))
                # print('difference between parent 2 and child 2', np.subtract(parent2, child2_genes))

                # append the children to the offspring list
                offspring.append(child1_genes)
                offspring.append(child2_genes)
            
            elif crossover_method == 'crossover_uniformsplit':
                # perform uniform crossover
                child1_genes = []
                child2_genes = []
                for i in range(len(parent1)):
                    x = random.random()
                    if x < 0.5:
                        child1_genes.append(parent1[i])
                        child2_genes.append(parent2[i])
                    else:
                        child1_genes.append(parent2[i])
                        child2_genes.append(parent1[i])
                
                offspring.append(np.array(child1_genes))
                offspring.append(np.array(child2_genes))
        
        
        # print('the offspring are' , offspring)
        return offspring
