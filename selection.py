import random
import numpy as np






def selection(self):
    ''' 
    Function to select the best agents from the population. 

    Methods are described here: http://www.iaeng.org/publication/WCE2011/WCE2011_pp1134-1139.pdf

    parameters:
    ----------------
        self: GeneticAlgorithm
            GeneticAlgorithm object
        population_fitness: list <float>
            List of fitness values of the population
        num_agents: int
            Number of agents to be selected

    returns:
    ----------------
        selected_population: list <float>
            List of indices of the selected agents from the original population

        '''

    population_fitness = [agent.fitness for agent in self.population]
    num_agents = self.num_select_agents

    if num_agents > len(population_fitness):
        raise ValueError('Number of agents to be selected is greater than the population size')


    selection_type = self.selection_type

    # Print selection type
    print('Selection type: {}'.format(selection_type))

    # initializing a new population
    selected_population = []
    
    if selection_type == 'tournament':
        """
        Tournament selection
        Selects the best agent from a random sample of agents.
        """
        population = np.array(list(enumerate(population_fitness)))

        for i in range(num_agents):
            # selecting a random sample of agents from the population
            tournament_size = 2 # binary tournament

            # choosing random agents from the population
            tournament = np.random.choice(range(len(population)), tournament_size, replace=True)
            tournament = population[tournament]

            # selecting the best agent from the tournament
            best_agent = np.max(tournament[:,1])

            # find index of best agent
            best_agent_id = tournament[tournament[:,1] == best_agent]

            # remove best agent from population
            population = population[population[:,1] != best_agent]

            # find index of best agent
            selected_population.append(int(best_agent_id[0][0]))



    elif selection_type == 'proportional-roulette-wheel':
        """
        In proportional roulette wheel, individuals are selected with a probability that is directly proportional to their 
        fitness values i.e. an individual‟s selection corresponds to a portion of a roulette wheel.
        """
        # Normalizing the fitness values to create a cumulative probability distribution
        fitness_sum = sum(population_fitness)
        norm_fitness = [fitness / fitness_sum for fitness in population_fitness]
        cum_fitness = np.cumsum(norm_fitness)
        # a

        # pass into roulette wheel function
        # selected_population = roulette_wheel(cum_fitness, num_agents)



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

        # List to store scaled ranks
        scaled_ranks = []
        scaled_indices = []

        for agent_index, fitness, rank in population_fitness:
            scaled_rank = calculate_scaled_rank(population_size, rank)

            # Ensuring indices and ranks match
            scaled_indices.append(agent_index)
            scaled_ranks.append(scaled_rank)

        # Normalizing the scaled ranks to create a probability distribution
        rank_sum = sum(scaled_ranks)
        scaled_ranks = [rank / rank_sum for rank in scaled_ranks]

        while len(selected_population) < num_agents:
            # Randomly selecting an agent from the population based on the scaled rank
            selected_agent = np.random.choice(scaled_indices, p=scaled_ranks)
            
            if selected_agent not in selected_population:
                selected_population.append(selected_agent)

    elif selection_type == 'elitism':
        """
        Elitism selection is a selection method where the best agents are always selected 
        for the next generation.
        """
        # Extracting the indices of agents
        numbered_fitness = list(enumerate(population_fitness))

        # Sorting the population by fitness
        sorted_fitness = sorted(numbered_fitness, key=lambda x: x[1], reverse=True)

        for i in range(num_agents):
            selected_population.append(sorted_fitness[i][0])

    # print('Population Fitness and indices: ({})'.format( list(enumerate(population_fitness))))
    print('Selected Population indices: ', selected_population)
    for i in selected_population:
        self.population[i].selected = True
    return self 
            
