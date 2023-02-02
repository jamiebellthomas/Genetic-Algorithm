from gym_examples.envs.grid_world import GridWorldEnv
from gymnasium.wrappers import flatten_observation

# Instantiate the environment
env = GridWorldEnv()

# Flatten the observation
env = flatten_observation(env)
