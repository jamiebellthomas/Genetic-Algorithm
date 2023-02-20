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
        self.GA = GeneticAlgorithm(
            environment='CartPole-v1',
            population_size=5,
            selection_type='elitism',
            fitness_sharing=True,
            sparse_reward=False,
            crossover_rate=0.7,
            mutation_rate=0.1,
            num_generations=1,
            parallel=0,
            plot=1,
            description='Parallel Test'
        )

    # Check output lengths
    def test_1(self):
        GA = self.GA
        GA.generation = 1
        GA = initialise_metrics(GA)
        GA, population_fitness, terminated = evaluate_fitness(GA)
        fitness_length = len(population_fitness)

        self.assertEqual(fitness_length, GA.population_size)
        
if __name__ == '__main__':
    unittest.main()
