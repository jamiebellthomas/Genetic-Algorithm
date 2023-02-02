import gymnasium as gym
import matplotlib.pyplot as plt 

env = gym.make('MountainCar-v0', render_mode="human")

# Observation and action space 
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))


# reset the environment and see the initial observation
obs = env.reset()
print("The initial observation is {}".format(obs))

# Sample a random action from the entire action space
random_action = env.action_space.sample()
print("The random action is {}".format(random_action))

# # Take the action and get the new observation space
new_obs, reward, terminated, truncated, info = env.step(random_action)
print("The new observation is {}".format(new_obs))

env.render()


import time 

# Number of steps you run the agent for 
num_steps = 1500

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    action = env.action_space.sample()
    
    # apply the action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.001)
    
    # If the epsiode is up, then start another one
    if terminated or truncated:
        env.reset()

# Close the env
env.close()