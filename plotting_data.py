import pandas as pd
import os
import plotly.graph_objects as go
import sys

# Plot metrics
def plot_metrics(self):
    """ Plot metrics
    This function plots the metrics recorded during the genetic algorithm.
    """
    # Plot metrics
    plot_all_fitness(self.metrics['all_fitness'], save_plot=self.run_tests, str_run_id=self.ID, str_model_folder=self.str_test_folder)

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
def plot_all_fitness(all_fitness, save_plot=False, str_run_id=None, str_model_folder=None):
    """ plot_all_fitness
    This function plots all the fitness scores recorded during the genetic algorithm.
    It plots the distribution of each generation.
    """
    # Convert all_fitness to list of strings to list of lists if strings
    if isinstance(all_fitness[0], str):
        all_fitness = list(map(lambda x: x[1:-1].split(','), all_fitness))
        all_fitness = [list(map(float, x)) for x in all_fitness]

    # Plot all fitness using plotly
    fig = go.Figure()
    for index, generation in enumerate(all_fitness):
        fig.add_trace(go.Scatter(x=[index+1]*len(generation),y=generation, name='Generation {}'.format(index+1), mode='markers'))

    # Plot mean fitness as a line
    mean_fitness = [sum(x)/len(x) for x in all_fitness]
    fig.add_trace(go.Scatter(x=list(range(1, len(mean_fitness)+1)), y=mean_fitness, name='Mean Fitness', mode='lines', line=dict(color='red')))

    # Plot best fitness as a dotted line
    best_fitness = [max(x) for x in all_fitness]
    fig.add_trace(go.Scatter(x=list(range(1, len(best_fitness)+1)), y=best_fitness, name='Best Fitness', mode='lines', line=dict(color='green', dash='dot')))

    # Set title
    fig.update_layout(title_text='Fitness Distribution')

    # Label axes
    fig.update_xaxes(title_text='Generation')
    fig.update_yaxes(title_text='Fitness')

    # Save figure
    if save_plot:
        # Create string for file path using str_model_folder and str_run_id
        str_file_path = str_model_folder + '/Test Graphs/{}_fitness_distribution.png'.format(str_run_id)
        # Save figure
        fig.write_image(str_file_path)
    else:
        fig.show()
    
# Plot metrics from file
if __name__ == '__main__':
    metrics = get_metrics_from_file(sys.argv[1])
    # metrics = get_metrics_from_file('0041')
    plot_all_fitness(metrics['all_fitness'].values, save_plot=False, str_run_id=None, str_model_folder=None)