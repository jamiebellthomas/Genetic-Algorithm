import time
import pandas as pd
import os
import json

from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model

from metrics import save_metrics
# from GeneticAlgorithm import GeneticAlgorithm


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
    class_variables['env'] = 'Environment object not saved'
    # TypeError: Object of type TimeLimit is not JSON serializable

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

    # Create a new folder for the model if it doesn't already exist if it exists, prompt the user to overwrite
    if os.path.exists(path):
        print('-------------------------------------------------')
        print('Model {} already exists. Create New Model, Overwrite Model or Discard New Model?'.format(ID))
        print('-------------------------------------------------')
        print('1. Create New Model')
        print('2. Overwrite Model')
        print('3. Discard New Model')
        print('-------------------------------------------------')

        user_input = input('Enter 1, 2 or 3: ')

        if user_input == '1':
            while os.path.exists(path):
                ID = int(ID) + 1
                ID = str(ID).zfill(4)
                path = 'Training/Saved Models/' + ID
            
            print('Creating new model {}...'.format(ID))
            os.mkdir(path)

        elif user_input == '2':
            print('Overwriting model {}...'.format(ID))  

        elif user_input == '3':
            print('Discarding new model...')
            exit()        

        else:
            print('Invalid input. Exiting...')
            exit()

    else:
        # Create a new folder for the model
        os.mkdir(path)

    # Concat the data into the dataframe
    df = pd.concat([df, pd.DataFrame([[ID, date, path, description]], columns=['ID', 'Date', 'Path', 'Description'])])

    # Save the dataframe to a csv file
    df.to_csv('Training/ModelDetails.csv', index=False)

    return path


def load_generation(path):
    """ 
    Load a generation of agents from a folder.
    
    parameters:
    ----------------
        path: str
            Path of the folder containing the models
    
    returns:
    ----------------
        class_settings: dict
            Dictionary of class variables for the GeneticAlgorithm object
        population: list
            List of NeuralNetwork objects
    """
    print('Loading models...')
    population = []

    class_settings = json.load(open(os.path.join(path, 'class_variables.json'), 'r'))
    
    # Load the models
    for filename in os.listdir(path):
        if filename.endswith('.h5'):
            print('Loading {}'.format(filename))
            model = load_model(os.path.join(path, filename))
            population.append(model)

    print('Models loaded from {}'.format(path))
    return class_settings, population

