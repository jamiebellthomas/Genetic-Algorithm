import unittest   # The test framework

from GeneticAlgorithm import GeneticAlgorithm
from metrics import initialise_metrics
from evaluate_fitness import evaluate_fitness
from selection import selection
from crossover import crossover
from mutation import mutate

class Test_TestSplitInput(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_TestSplitInput, self).__init__(*args, **kwargs)
        # default
        self.GA = GeneticAlgorithm(
            environment='CartPole-v1',
            population_size = 10
        )
        GA = self.GA
        GA.generation = 1
        num_agents = 4
        GA = initialise_metrics(GA)
        GA, population_fitness, _ = evaluate_fitness(GA)
        selected_population = selection(GA, population_fitness, num_agents)
        offspring = crossover(GA, selected_population)
        self.population_size = GA.population_size
        self.population_fitness = population_fitness
        self.selected_population = selected_population
        self.offspring = offspring
        self.num_agents = num_agents


    # Check output lengths
    def test_1(self):
        fitness_length = len(self.population_fitness)
        self.assertEqual(fitness_length, GA.population_size)


    def test_2(self):
        selected_length = len(self.selected_population)
        self.assertEqual(selected_length, self.num_agents)
    
    def test_3(self):
        offspring_length = len(self.offspring)
        self.assertEqual(offspring_length, self.GA.population_size)
        

if __name__ == '__main__':
    unittest.main()
