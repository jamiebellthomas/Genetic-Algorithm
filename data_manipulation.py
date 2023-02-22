import time
import pandas as pd
import os
import json
import gym

import tensorflow as tf
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')


from NeuralNetwork import NeuralNetwork
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


def load_generation_details(model_ID):
    """
    Load the json file containing the details of the models.
    
    parameters:
    ----------------
        model_ID: str
            ID of the model

    returns:
    ----------------
        class_settings: dict
            Dictionary of class variables for the GeneticAlgorithm object
    """
    path = 'Training/Saved Models/' + model_ID

    # Load the json file
    with open(os.path.join(path, 'class_variables.json'), 'r') as f:
        class_settings = json.load(f)

    return class_settings


def load_generation_population(model_ID, index=None):
    """ 
    Load a generation of agents from a folder.
    
    parameters:
    ----------------
        path: str
            Path of the folder containing the models
    
    returns:
    ----------------
        population: list
            List of NeuralNetwork objects
        index: int
            Index of specific agent in the population to load
    """
    print('Loading models...')
    path = 'Training/Saved Models/' + model_ID

    population = []

    if index is None:
        # Load the models
        for filename in os.listdir(path):
            if filename.endswith('.h5'):
                print('Loading {}'.format(filename))
                model = load_model(os.path.join(path, filename))
                population.append(model)
    else:
        filename = 'Agent_{}.h5'.format(index)
        model = load_model(os.path.join(path, filename))
        population.append(model)

    print('Models loaded from {}'.format(path))
    return population


def load(model_ID, total_population=False, render=True):
    """
    Loads population of agents from a folder.

    parameters:
    ----------------
        model_ID: str
            ID of the model
        total_population: bool
            Whether to load the entire population or just the best agent
        render: bool
            Whether to render the environment

    returns:
    ----------------
        population: list
            List of NeuralNetwork objects
    """
    # Load the class variables
    class_settings = load_generation_details(model_ID)

    if total_population:
        population_indices = load_generation_population(model_ID)
        population = []
        for model in population_indices:
            population.append(NeuralNetwork(4, 2, model=model))
        agent = population[class_settings['best_agent']]

    else:
        top_index = class_settings['best_agent']
        top_agent = load_generation_population(model_ID, top_index)
        population = []
        population.append(NeuralNetwork(4, 2, model=top_agent[0]))
        agent = population[0]


    if render:
        env = gym.make(class_settings['environment'], render_mode="human")
        observation = env.reset()
        env.render()

        if type(observation) == tuple:
            observation = observation[0]

        done = False
        while not done:
            # Get action from agent and pass it to the environment
            action = agent.predict_action(observation)
            # Try with 4 outputs if it errors except with 5 outputs this will depend on version of gym
            try:
                observation, reward, done, info = env.step(action)
            except:
                observation, reward, done, truncation, info = env.step(action)
            
            # Render the environment
            env.render()
            
            if truncation:
                break

        env.close()
