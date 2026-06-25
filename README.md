DQN CartPole Agent (PyTorch)

This project implements a Deep Q-Network (DQN) agent that learns to solve the CartPole-v1 environment using reinforcement learning.

The agent learns to balance a pole on a cart by interacting with the environment and optimizing a neural network policy.

Overview

The agent uses a Deep Q-Network to estimate Q-values for each possible action given a state.

It is trained using experience replay and a target network to stabilize learning.

Environment

- Gymnasium CartPole-v1 environment
- State space: 4 continuous values
- Action space: 2 discrete actions (left or right)

Model Architecture

The Q-network is a fully connected neural network:

- Input layer: state vector
- Two hidden layers with ReLU activation
- Output layer: Q-values for each action

Training Method

The agent is trained using:

- Experience replay buffer
- Epsilon-greedy exploration strategy
- Target network for stable Q-learning updates
- Mean Squared Error loss between predicted Q-values and target Q-values

Key Concepts Used

- Temporal Difference Learning
- Bellman Equation
- Experience Replay
- Target Network Stabilization
- Exploration vs Exploitation tradeoff

Training Process

The agent interacts with the environment in episodes:

- Selects action using epsilon-greedy policy
- Stores transitions in replay buffer
- Samples random batches for training
- Updates policy network using gradient descent
- Periodically updates target network

Evaluation

After training, the agent is evaluated by running a full episode using the learned policy without exploration.

The agent selects actions greedily based on learned Q-values.

Limitations

This implementation is a basic DQN and has limitations:

- No Double DQN (overestimation bias possible)
- No Prioritized Experience Replay
- No reward shaping improvements
- Sensitive to hyperparameters
- May require many episodes for stable convergence
