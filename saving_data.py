import time
import pandas as pd
import os

from tensorflow.keras.models import save_model


def save_generation(population, description):
    """ 
    Save generation of agents to a folder.
    Possible addition: incorporate a checkpoint to save models during training in case computer says no.
    
    parameters:
    ----------------
        population: list <NeuralNetwork>
            List of NeuralNetwork objects representing the population
        description: str
            Description of the model. This should include at least the environment name,
            the number of generations, the population size. Any other hyperparameters
            required to reproduce the model should also be included.
     """
    print('Saving models...')
    path = update_model_details(description)

    # Save the models
    for i, agent in enumerate(population):
        filename = 'Agent_{}.h5'.format(i)
        file_path = os.path.join(path, filename)

        save_model(agent.model, file_path)
        print(len(population))

    print('Models saved to {}'.format(path))




def update_model_details(description):
    """ 
    Function appends model details to the ModelDetails.csv file. It also creates a 
    new folder for the agents to be saved to.
    
    parameters:
    ----------------
        description: str
            Description of the model. This should include at least the environment name,
            the number of generations, the population size. Any other hyperparameters 
            required to reproduce the model should also be included.

    returns:
    ----------------
        path: str
            Path of the model
    """
    # Read the csv file
    df = pd.read_csv('Training/ModelDetails.csv')
    
    # Extract the last ID and increment it by 1 to get the new ID
    ID = int(df['ID'].iloc[-1]) + 1
    ID = str(ID).zfill(4)

    # Model details
    date = time.strftime("%Y-%m-%d")
    path = 'Training/Saved Models/' + ID

    # Create a new folder for the model
    os.mkdir(path)

    # Concat the data into the dataframe
    df = pd.concat([df, pd.DataFrame([[ID, date, path, description]], columns=['ID', 'Date', 'Path', 'Description'])])

    # Save the dataframe to a csv file
    df.to_csv('Training/ModelDetails.csv', index=False)

    return path
