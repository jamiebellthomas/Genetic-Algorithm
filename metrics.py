import pandas as pd
import os

def initialise_metrics(self):
    """ Initialise metrics
    This function initialises the metrics to be used in the genetic algorithm.
    """
    # Initialise metrics
    self.metrics = {}
    self.metrics['generation'] = []
    self.metrics['best_fitness'] = []
    self.metrics['mean_fitness'] = []
    self.metrics['best_agent'] = []
    self.metrics['duration'] = []
    self.metrics['best_agent'] = None

    return self


def update_metrics(self):
    """ Update metrics
    This function updates the metrics to be used in the genetic algorithm.
    """
    # Update metrics
    self.metrics['generation'].append(self.generation)
    self.metrics['best_fitness'].append(self.best_fitness)
    self.metrics['mean_fitness'].append(self.mean_fitness)
    self.metrics['duration'].append(self.duration)
    self.metrics['best_agent'] = self.best_agent

    return self


def save_metrics(self, path):
        """ Save metrics
        This function saves the metrics recorded during the genetic algorithm as a csv file.
        """
        # Save metrics
        metrics = pd.DataFrame(self.metrics)
        filename = os.path.join(path, 'metrics.csv')
        metrics.to_csv(filename, index=False)