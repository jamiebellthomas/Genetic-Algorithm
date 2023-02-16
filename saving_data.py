import time
import pandas as pd
import os
import json

from tensorflow.keras.models import save_model

from metrics import save_metrics


def save_generation(self):
    """ 
    Save generation of agents to a folder.
    Possible addition: incorporate a checkpoint to save models during training in case computer says no.
    
    parameters:
    ----------------
        self: GeneticAlgorithm object
            GeneticAlgorithm object
     """
    print('Saving models...')
    path = update_model_details(self.description)

    # Save the models
    for i, agent in enumerate(self.population):
        filename = 'Agent_{}.h5'.format(i)
        file_path = os.path.join(path, filename)

        save_model(agent.model, file_path)
        print(len(self.population))

    # Save class variables for reproducibility. Format as json file.
    class_variables = self.__dict__
    class_variables['population'] = 'Population saved as h5 files'

    with open(os.path.join(path, 'class_variables.json'), 'w') as f:
        # Format the json file to be human readable, with 4 spaces per indent
        json.dump(class_variables, f, indent=4)

    # Save metrics
    save_metrics(self, path)
    
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

