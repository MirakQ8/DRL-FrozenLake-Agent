# FROZEN-LAKE-AGENT

This project features a Deep Q-Learning (DQN) agent that has been implemented and trained to successfully solve the Frozen Lake environment from OpenAI's Gym. The agent leverages a neural network to approximate Q-values, enabling it to learn optimal policies through experience and interaction with the environment. The primary goal is to navigate the slippery gridworld, avoid holes, and reach the goal state using reinforcement learning techniques.
## Files

- Trained model: `frozen_lake_dqn`
- Script: `FrozenAgent.py`

## Usage

1. Download both the trained model and the script.
2. In `FrozenAgent.py`, set the correct path to the model file.

Example:

```python
model_path = r"YOUR/PATH/TO/frozen_lake_dqn"
