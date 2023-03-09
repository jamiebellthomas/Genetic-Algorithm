# This script will read in metrics from multiple metric csv files and plot them on the same graph.
# Copy test models from their original directory to the directory called MultiPlot.
# You can plot up to 4 models on the same graph.
# Alot of the code is based of plotting_data.py, just reformatted so multiple models can be plotted quickly on the same graph. 

import pandas as pd
import os
import plotly.graph_objects as go
import sys

# Read how many models to plot from the MultiPlot directory

num_models = len(os.listdir('MultiPlot'))
def extract_model_ids():
    """ Extract model ids
    This function extracts the model ids from the MultiPlot directory.
    """
    # Get list of model ids
    model_ids = os.listdir('MultiPlot')
    # Remove 'Graphs' folder if it exists
    if 'Graphs' in model_ids:
        model_ids.remove('Graphs')
    return model_ids
def extract_metrics_from_file(str_run_id):
    """ Get metrics from file
    This function gets the metrics recorded during the genetic algorithm from a csv file.
    """
    # Get path to metrics file
    path = os.path.join('MultiPlot', str_run_id, 'metrics.csv')

    # Read metrics from file
    metrics = pd.read_csv(path)

    return metrics

def combine_metrics(metrics_list):
    """This concatenates all the metrics into a dictionary where each model id is a key and the value is the metrics for that model
    """
    combined_metrics = {}
    for index, model_id in enumerate(extract_model_ids()):
        combined_metrics[model_id] = metrics_list[index]
    return combined_metrics

def plot_all_fitness(combined_metrics, model_ids):
    """
    Reads in the combined metrics dictionary and plots each of them on the same graph.
    Each model will be plotted with a different colour.
    """
    colors = ['red', 'blue', 'green', 'orange']
    fig = go.Figure()
    for index, model_id in enumerate(model_ids):
        all_fitness = combined_metrics[model_id]['all_fitness'].values
        all_fitness = list(map(lambda x: x[1:-1].split(','), all_fitness))
        all_fitness = [list(map(float, x)) for x in all_fitness]
        #for fitness_index, generation in enumerate(all_fitness):
        #    fig.add_trace(go.Scatter(x=[fitness_index+1]*len(generation),y=generation, name=f'Generation {fitness_index+1} ({model_id})', mode='markers', marker_color=colors[index]))
        # Plot mean fitness as a line
        mean_fitness = [sum(x)/len(x) for x in all_fitness]
        fig.add_trace(go.Scatter(x=list(range(1, len(mean_fitness)+1)), y=mean_fitness, name=f'Mean Fitness ({model_id})', mode='lines', line=dict(color=colors[index])))
        # Plot best fitness as a dotted line
        #best_fitness = [max(x) for x in all_fitness]
        #fig.add_trace(go.Scatter(x=list(range(1, len(best_fitness)+1)), y=best_fitness, name=f'Best Fitness ({model_id})', mode='lines', line=dict(color=colors[index], dash='dot')))
    fig.update_layout(xaxis_title='Generation', yaxis_title='Fitness')
    # increase the size of units
    fig.update_layout(font=dict(size=30))
    # Make axis labels larger
    fig.update_xaxes(title_font=dict(size=30))
    fig.update_yaxes(title_font=dict(size=30))
    
    # 
    fig.update_layout(
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    # Make legend larger
    fig.update_layout(legend=dict(font_size=30))
    return fig
def save_fig(fig, filename):
    """ Save figure
    This function saves the figure to a file.
    """
    # Save figure
    fig.write_image(f'MultiPlot/Graphs/{filename}', width=1920, height=1080, scale=6)


metrics_list = [extract_metrics_from_file(x) for x in extract_model_ids()]
combined_metrics = combine_metrics(metrics_list)
fig = plot_all_fitness(combined_metrics, extract_model_ids())
save_fig(fig, 'combined_fitness.png')
