from tensorflow.keras.models import save_model


def save_agent(model, env, generation, version):
    """ 
    Save model function. 
    Possible addition: incorporate a checkpoint to save models during training in case computer says no.
    
    parameters:
    ----------------
        model: keras model
            Agent model to be saved
        env: str
            Environment name 
        generation: int
            Generation number
        version: int
            Version number
     """
    print('Saving model...')
    save_model(model, 'Training/Saved Models/{}_{}_{}.h5'.format(env, generation, version))
