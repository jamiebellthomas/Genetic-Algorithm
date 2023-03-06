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
    self.metrics['total duration'] = []
    self.metrics['all_fitness'] = []
    self.metrics['best_agent'] = []
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
    self.metrics['all_fitness'].append(self.all_fitness)
    self.metrics['best_agent'].append(self.best_agent)

    # Update total duration
    if len(self.metrics['total duration']) == 0:
        self.metrics['total duration'].append(self.duration)
    else:
        self.metrics['total duration'].append(self.duration + self.metrics['total duration'][-1])

    return self


def save_metrics(self, path):
        """ Save metrics
        This function saves the metrics recorded during the genetic algorithm as a csv file.
        """
        # Save metrics
        metrics = pd.DataFrame(self.metrics)
        filename = os.path.join(path, 'metrics.csv')
        metrics.to_csv(filename, index=False)