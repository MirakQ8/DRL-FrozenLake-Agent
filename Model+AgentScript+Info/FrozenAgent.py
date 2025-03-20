import tensorflow as tf
import numpy as np
import gymnasium as gym
import time

# Define the correct path
model_path = r"C:\Users\XOX\Downloads\frozen_lake_dqn.h5"

loaded_model = tf.keras.models.load_model(model_path, compile=False)

env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=True)

# Convert state to one-hot encoding
def one_hot_state(state, num_states=16):
    state_one_hot = np.zeros(num_states)
    state_one_hot[state] = 1
    return state_one_hot.reshape(1, -1)

# Run the trained agent
def run_agent(model, num_episodes=5, sleep_time=0.5):
    for episode in range(num_episodes):
        state = env.reset()[0]
        state = one_hot_state(state)
        done = False
        total_reward = 0

        print(f"Episode {episode + 1} started")

        while not done:
            env.render()
            time.sleep(sleep_time)

            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = one_hot_state(next_state)

            total_reward += reward

            if done:
                env.render()
                time.sleep(sleep_time)
                print(f"Episode {episode + 1} ended with reward: {total_reward}")

    env.close()

# Run the agent
run_agent(loaded_model)
