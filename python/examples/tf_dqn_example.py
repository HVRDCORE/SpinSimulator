#!/usr/bin/env python3
"""
Example of using Deep Q-Network (DQN) with the poker simulator.
This demonstrates how to train a reinforcement learning agent to play poker
using TensorFlow and the RLEnvironment interface.
"""
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from collections import deque
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from typing import List, Tuple, Dict, Any, Union

# Add the parent directories to the path so we can import our module
examples_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(examples_dir)
project_dir = os.path.dirname(python_dir)
sys.path.extend([python_dir, project_dir])

# Import the poker module
try:
    import poker_core as pc
    from poker.rl_interface import RLEnvironment
except ImportError:
    print("Attempting to import the module from various locations...")
    # Try to find the module
    for path in [project_dir, python_dir, examples_dir]:
        for file in os.listdir(path):
            if file.startswith('poker_core') and file.endswith('.so'):
                print(f"Found module at {os.path.join(path, file)}")
                sys.path.insert(0, path)

    # Try importing again
    import poker_core as pc
    from poker.rl_interface import RLEnvironment

class DQNAgent:
    """
    Deep Q-Network agent for poker.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 memory_size: int = 100000, batch_size: int = 64,
                 gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            memory_size: Size of replay memory
            batch_size: Batch size for training
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            learning_rate: Learning rate for the optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self) -> models.Model:
        """
        Build a neural network model for DQN.
        
        Returns:
            A Keras Model
        """
        model = models.Sequential()
        # Input layer
        model.add(layers.Dense(256, input_dim=self.state_size, activation='relu'))
        # Hidden layers
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        # Output layer
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        # Compile the model
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """
        Update the target model with the weights of the main model.
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, legal_actions: List[int] = None) -> int:
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            legal_actions: List of legal actions (if None, all actions are legal)
            
        Returns:
            The chosen action
        """
        # With probability epsilon, choose a random action (exploration)
        if np.random.rand() <= self.epsilon:
            if legal_actions:
                return random.choice(legal_actions)
            else:
                return random.randrange(self.action_size)
        
        # Otherwise, choose the action with the highest Q-value (exploitation)
        act_values = self.model.predict(np.array([state]), verbose=0)[0]
        
        # Filter legal actions
        if legal_actions:
            # Create a mask for illegal actions
            mask = np.ones(self.action_size) * float('-inf')
            for a in legal_actions:
                mask[a] = 0
            # Apply mask
            act_values = act_values + mask
        
        return np.argmax(act_values)
    
    def replay(self, batch_size: int = None):
        """
        Train the model using experience replay.
        
        Args:
            batch_size: Batch size for training (defaults to self.batch_size)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # If not enough samples in memory, return
        if len(self.memory) < batch_size:
            return
        
        # Sample a random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            # Calculate target Q-value
            if done:
                target = reward
            else:
                # Double DQN: Use main network to select action, target network to compute Q-value
                next_action = np.argmax(self.model.predict(np.array([next_state]), verbose=0)[0])
                target = reward + self.gamma * \
                    self.target_model.predict(np.array([next_state]), verbose=0)[0][next_action]
            
            # Update Q-value for the chosen action
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            
            # Train the network
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name: str):
        """
        Load model weights from a file.
        
        Args:
            name: File name
        """
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name: str):
        """
        Save model weights to a file.
        
        Args:
            name: File name
        """
        self.model.save_weights(name)

def train_dqn_agent(env: RLEnvironment, agent: DQNAgent, 
                  episodes: int = 1000, target_update_freq: int = 100,
                  batch_size: int = 32, max_steps: int = 200,
                  render_freq: int = 100, save_freq: int = 100,
                  save_dir: str = "models"):
    """
    Train the DQN agent on the poker environment.
    
    Args:
        env: RLEnvironment instance
        agent: DQNAgent instance
        episodes: Number of episodes to train
        target_update_freq: Frequency of target model updates
        batch_size: Batch size for training
        max_steps: Maximum steps per episode
        render_freq: Frequency of rendering
        save_freq: Frequency of saving the model
        save_dir: Directory to save the model
    
    Returns:
        List of episode rewards
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # List to store episode rewards
    episode_rewards = []
    
    # Training loop
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        # Episode loop
        while not done and step < max_steps:
            # Get legal actions
            legal_actions = []
            for i in range(agent.action_size):
                # This is a simplified way to check legal actions and should be replaced
                # with a proper method from the environment
                try:
                    _, _, _, _ = env.step(i)
                    legal_actions.append(i)
                except:
                    pass
            
            # Choose an action
            action = agent.act(state, legal_actions)
            
            # Take a step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            step += 1
            
            # Train the agent
            agent.replay(batch_size)
            
            # Render the environment
            if episode % render_freq == 0:
                env.render()
                print(f"Episode: {episode}/{episodes}, Step: {step}, Action: {action}, Reward: {reward}")
        
        # Update target model
        if episode % target_update_freq == 0:
            agent.update_target_model()
            print(f"Target model updated at episode {episode}")
        
        # Save the model
        if episode % save_freq == 0:
            agent.save(os.path.join(save_dir, f"dqn_agent_episode_{episode}.h5"))
            print(f"Model saved at episode {episode}")
        
        # Add episode reward to list
        episode_rewards.append(total_reward)
        
        # Print episode info
        print(f"Episode: {episode}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
    
    # Save final model
    agent.save(os.path.join(save_dir, "dqn_agent_final.h5"))
    print("Training completed. Final model saved.")
    
    return episode_rewards

def evaluate_agent(env: RLEnvironment, agent: DQNAgent, episodes: int = 100):
    """
    Evaluate the trained DQN agent.
    
    Args:
        env: RLEnvironment instance
        agent: DQNAgent instance
        episodes: Number of evaluation episodes
        
    Returns:
        Average reward per episode
    """
    # Turn off exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    # List to store episode rewards
    episode_rewards = []
    
    # Evaluation loop
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Choose an action
            action = agent.act(state)
            
            # Take a step
            next_state, reward, done, info = env.step(action)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Render the environment
            env.render()
        
        # Add episode reward to list
        episode_rewards.append(total_reward)
        
        # Print episode info
        print(f"Evaluation Episode: {episode}/{episodes}, Total Reward: {total_reward}")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Calculate average reward
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Evaluation completed. Average Reward: {avg_reward}")
    
    return avg_reward

def main():
    """
    Main function to train and evaluate a DQN agent on the poker environment.
    """
    # Create environment
    env = RLEnvironment(num_players=3, initial_stack=1000, small_blind=10, big_blind=20)
    
    # Get state and action dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        memory_size=100000,
        batch_size=64,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001
    )
    
    # Train agent
    train_episodes = 1000
    print(f"Training agent for {train_episodes} episodes...")
    episode_rewards = train_dqn_agent(
        env=env,
        agent=agent,
        episodes=train_episodes,
        target_update_freq=100,
        batch_size=32,
        max_steps=200,
        render_freq=100,
        save_freq=100,
        save_dir="models/dqn"
    )
    
    # Plot training rewards
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig("training_rewards.png")
        plt.show()
    except ImportError:
        print("matplotlib not available for plotting.")
    
    # Evaluate agent
    print("Evaluating agent...")
    avg_reward = evaluate_agent(
        env=env,
        agent=agent,
        episodes=100
    )
    
    print(f"Final average reward: {avg_reward}")

if __name__ == "__main__":
    main()