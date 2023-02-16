import pandas as pd
import matplotlib.pyplot as plt
import os

# Plot metrics
def plot_metrics(self):
    """ Plot metrics
    This function plots the metrics recorded during the genetic algorithm.
    """
    # Plot metrics
    metrics = pd.DataFrame(self.metrics)

# Get metrics from file
def get_metrics_from_file(str_run_id):
    """ Get metrics from file
    This function gets the metrics recorded during the genetic algorithm from a csv file.
    """
    # Get path to metrics file
    path = os.path.join('Training', 'Saved Models', str_run_id, 'metrics.csv')

    # Read metrics from file
    metrics = pd.read_csv(path)

    return metrics

# Plot metrics from file
def plot_all_fitness(metrics):
    """ plot_all_fitness
    This function plots all the fitness scores recorded during the genetic algorithm.
    It plots the distribution of each generation.
    """
    # Plot 10 points for each generation
    num_points = 10
    
    
# Plot metrics from file
if __name__ == '__main__':
    metrics = get_metrics_from_file('0036')
    plot_all_fitness(metrics)