import random
import numpy as np



def selection(GA, selection_type, population_fitness):
        ''' 
        Function to select the best agents from the population. 

        Methods are described here: http://www.iaeng.org/publication/WCE2011/WCE2011_pp1134-1139.pdf

        parameters:
        ----------------
            GA: GeneticAlgorithm
                GeneticAlgorithm object
            selection_type: str
                Type of selection. Possible values: 'roulette_wheel', 'tournament'
            population_fitness: list <float>
                List of fitness values of the population

        returns:
        ----------------
            selected_population: list <float>
                List of indices of the selected agents from the original population

        '''
        # initializing a new population
        selected_population = []
        
        if selection_type == 'tournament':
            """
            Tournament selection
            Selects the best agent from a random sample of agents.
            """
            # chosen size to be around 20% of the population - can be changed
            def selection(self, t_size=len(GA.population)//5):
                # selecting a random sample of indexes of agents 
                selected_agents = random.sample(range(len(self.population)), t_size)

                # selecting the best agent from the random sample
                selected_agent = max(selected_agents, key=lambda x: self.population_fitness[x])

                return selected_agent
            
            for _ in range(len(GA.population)):
                selected_population.append(selection(GA))

            # a list of indices of the selected agents from the original population
            return selected_population


        elif selection_type == 'proportional-roulette-wheel':
            """
            In proportional roulette wheel, individuals are selected with a probability that is directly proportional to their 
            fitness values i.e. an individual‟s selection corresponds to a portion of a roulette wheel.
            """
            # Normalizing the fitness values to create a probability distribution
            fitness_sum = sum(population_fitness)
            probabilities = [fitness / fitness_sum for fitness in population_fitness]

            # generating a random number between 0 and 1 for the population selection
            random_number = np.random.rand()

            # selecting the population based on the random number
            def selection(self, random_number=random_number):
                for index, prob in enumerate(self.selection_probs):
                    random_number -= prob
                    # all the agents with a probability greater than the random number are selected
                    if random_number <= 0:
                        selected_population.append(index)

                    return selected_population

            selected_population = selection(GA)

            # a list of indices of the selected agents from the original population
            return selected_population


        elif selection_type == 'rank-based-rolette-wheel':
            '''
            Rank-based roulette wheel selection is the selection strategy where the probability of
            a chromosome being selected is based on its fitness rank relative to the entire population. 
            Rank-based selection schemes first sort individuals in the population according to their 
            fitness and then computes selection probabilities according to their ranks rather than 
            fitness values.

            Scaled ranks are calculated using the following formula:

            scaled_rank = 2 - selection_pressure + (2 * (selection_pressure - 1) * (rank - 1) / (population_size - 1))
            '''
            # Create a list of tuples containing the index, fitness, and rank of each agent
            # Index corresponds to the index of the agent in the population
            # Fitness corresponds to the fitness of the agent
            # Rank corresponds to the rank of the agent in the population
            population_fitness = list(enumerate(population_fitness))
            population_fitness = sorted(population_fitness, key=lambda x: x[1])
            population_fitness = [(index, fitness, rank) for rank, (index, fitness) in enumerate(population_fitness)]


            def calculate_scaled_rank(population_size, rank, selection_pressure=2):
                ''' 
                Function to calculate the scaled rank of an agent. Method is outlined in the docstring of selection function.
                The larger the selection pressure, the more extreme the scaling will be.
                
                '''
                rank += 1
                scaled_rank = 2 - selection_pressure + (2 * (selection_pressure - 1) * (rank - 1) / (population_size - 1))
                return scaled_rank

            # Calculate the scaled rank of each agent
            # population_size = GA.population_size
            population_size = len(population_fitness)

            # New list created to prevent overwriting of population_fitness during iteration
            population_fitness_ = []

            # Creat cumulative distribution of scaled ranks
            cumulative_distribution = []
            cumulative_value = 0

            for agent_index, fitness, rank in population_fitness:
                scaled_rank = calculate_scaled_rank(population_size, rank)
                population_fitness_.append((agent_index, fitness, rank, scaled_rank))

                cumulative_value += scaled_rank
                cumulative_distribution.append((agent_index, cumulative_value))

            # Selecting agents from the population
            selected_population = roulette_wheel(cumulative_distribution, num_agents=2)

            return selected_population
            



def roulette_wheel(cumulative_distribution, num_agents=1):
    '''
    Function to randomly select an agent from the population based on the cumulative fitness distribution.

    parameters:
    ----------------
        cumulative_distribution: list <float>
            List of cumulative fitness values of the population
        num_agents: int
            Number of agents to be selected
    
    returns:
    ----------------
        selected_agents: list <int>
            List of indices of the selected agents from the original population
    '''
    print('Initializing Roulette Wheel Selection')
    # The total fitness of the population
    _, total_fitness = cumulative_distribution[-1]

    # Randomly selecting agents from the population according to distribution
    selected_agents = []

    # Selecting agents until the desired number of agents is selected
    while len(selected_agents) < num_agents:
        print('Total Fitness: ', total_fitness)
        # Generating a random number between 0 and the total fitness
        random_number = np.random.rand() * total_fitness

        for agent_index, cumulative_value in cumulative_distribution:
            print('Random Number: ', random_number)
            print('Cumulative Value: ', cumulative_value)
            # Select agent that has a cumulative fitness value greater than the random number and is not already selected
            if random_number < cumulative_value and agent_index not in selected_agents:
                selected_agents.append(agent_index)

    return selected_agents
                

