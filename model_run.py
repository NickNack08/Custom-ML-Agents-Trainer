import os
import time
import numpy as np
import tensorflow as tf
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Paths to the saved models
model_dir = r'C:\Users\nkd2840\UNITY\Custom\results'
actor_save_path = os.path.join(model_dir, 'actor_model.h5')
critic_save_path = os.path.join(model_dir, 'critic_model.h5')

# Load the saved models
actor = tf.keras.models.load_model(actor_save_path)
critic = tf.keras.models.load_model(critic_save_path)

# Specify the path to the Unity environment executable
file_name = 'Custom'

# Set the arguments for headless mode (remove these arguments if you want to visualize in Unity Editor)
unity_args = []

# Create the EngineConfigurationChannel
channel = EngineConfigurationChannel()

# Create the Unity environment with the specified arguments and side channel
env = UE(file_name=file_name, seed=1, side_channels=[channel], additional_args=unity_args)

# Set the time scale 
channel.set_configuration_parameters(time_scale=1.0)

# Reset the environment
env.reset()

# Get behavior name
behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior: {behavior_name}")
spec = env.behavior_specs[behavior_name]

# Define the number of episodes to visualize
num_episodes = 10

# Main loop to run and visualize the model
for episode in range(num_episodes):
    start_time = time.time()
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1  # -1 indicates not yet tracking
    done = False

    while not done:
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]

        state = decision_steps[tracked_agent].obs[0]

        # Predict the action using the loaded actor model
        action_probs = actor(np.array([state]))
        action = tf.random.categorical(tf.math.log(action_probs), num_samples=1)[0, 0].numpy()

        action_tuple = ActionTuple(discrete=np.array([[action]]))
        env.set_actions(behavior_name, action_tuple)
        
        # Take a step in the environment
        env.step()

        # Get the next state and reward
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        if tracked_agent in decision_steps:
            next_state = decision_steps[tracked_agent].obs[0]
            reward = decision_steps[tracked_agent].reward
            done = False
        else:
            next_state = terminal_steps[tracked_agent].obs[0]
            reward = terminal_steps[tracked_agent].reward
            done = True

    episode_end_time = time.time()
    print(f"Episode {episode} completed in {episode_end_time - start_time:.2f} seconds, Reward: {reward}")

    

env.close()
print("Closed environment")