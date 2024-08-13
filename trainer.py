import time
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
import tensorflow as tf
import wandb

# Initialize wandb
wandb.init(project="unity-ml-agents", name="roller-ball-training")

# Specify the path to the Unity environment executable
file_name = 'Custom'

# Set the arguments for headless mode
unity_args = ["-batchmode", "-nographics"]
# unity_args = []
# Create the EngineConfigurationChannel
channel = EngineConfigurationChannel()

# Create the Unity environment with the specified arguments and side channel
env = UE(file_name=file_name, seed=1, side_channels=[channel], additional_args=unity_args)

# Set the time scale 
channel.set_configuration_parameters(time_scale=2.0)
# Reset the environment
env.reset()

# Get behavior name
behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior: {behavior_name}")
spec = env.behavior_specs[behavior_name]

# Define the actor and critic networks
actor = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(spec.action_spec.discrete_branches[0], activation='softmax'),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(spec.action_spec.discrete_branches[0], activation='softmax')
])

critic = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1),
    # tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dense(1)
])

# Define optimizer and loss functions
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Main training loop
num_episodes = 500
gamma = 0.99

# Initialize cumulative metrics and episode rewards list
cumulative_reward = 0
cumulative_actor_loss = 0
cumulative_critic_loss = 0
total_steps = 0
episode_rewards = []
train_start_time = time.time()
for episode in range(num_episodes):
    start_time = time.time()
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1  # -1 indicates not yet tracking
    episode_reward = 0
    episode_actor_loss = 0
    episode_critic_loss = 0
    step_count = 0
    done = False

    while not done:
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]

        state = decision_steps[tracked_agent].obs[0]

        with tf.GradientTape(persistent=True) as tape:
            action_probs = actor(np.array([state]))
            action = tf.random.categorical(tf.math.log(action_probs), num_samples=1)[0, 0]
            
            action_tuple = ActionTuple(discrete=np.array([[action]]))
            env.set_actions(behavior_name, action_tuple)
            
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if tracked_agent in decision_steps:
                next_state = decision_steps[tracked_agent].obs[0]
                reward = decision_steps[tracked_agent].reward
                done = False
            else:
                next_state = terminal_steps[tracked_agent].obs[0]
                reward = terminal_steps[tracked_agent].reward
                done = True

            # print(f"Step {step_count}: Reward = {reward:.2f}, Cumulative Reward = {episode_reward + reward:.2f}")

            state_value = critic(np.array([state]))[0, 0]
            next_state_value = critic(np.array([next_state]))[0, 0] if not done else 0
            advantage = reward + gamma * next_state_value - state_value

            actor_loss = -tf.math.log(action_probs[0, action]) * advantage
            critic_loss = tf.square(advantage)

            actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
            critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        
        episode_reward = reward
        episode_actor_loss += actor_loss.numpy()
        episode_critic_loss += critic_loss.numpy()
        step_count += 1

        # Update cumulative metrics
        cumulative_reward += reward
        cumulative_actor_loss += actor_loss.numpy()
        cumulative_critic_loss += critic_loss.numpy()
        total_steps += 1


        # Log cumulative metrics over steps
        wandb.log({
            "Cumulative Reward": cumulative_reward,
            "Cumulative Actor Loss": cumulative_actor_loss,
            "Cumulative Critic Loss": cumulative_critic_loss,
            "Total Steps": total_steps
        })
    print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")

    episode_end_time = time.time()

    # Add the episode reward to the list
    episode_rewards.append(episode_reward)

    # Compute and log average reward over every 20 episodes
    if (episode + 1) % 20 == 0:
        avg_reward = np.mean(episode_rewards[-20:])
        wandb.log({
            "Average Reward (Last 20 episodes)": avg_reward
        })

    # Log episode-level metrics
    wandb.log({
        "Episode": episode,
        "Episode Reward": episode_reward,
        "Episode Actor Loss": episode_actor_loss,
        "Episode Critic Loss": episode_critic_loss,
        "Episode Duration": episode_end_time - start_time
    })
train_end_time = time.time()

# Save the trained models
actor_save_path = r'C:\Users\nkd2840\UNITY\Custom\results\actor_model.h5'
critic_save_path = r'C:\Users\nkd2840\UNITY\Custom\results\critic_model.h5'

actor.save(actor_save_path)
critic.save(critic_save_path)

print(f"Saved actor model to {actor_save_path}")
print(f"Saved critic model to {critic_save_path}")

env.close()
print(f"trainign took {train_end_time - train_start_time} Seconds")
print("Closed environment")