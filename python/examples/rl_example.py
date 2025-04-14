import os
import sys
import numpy as np
import time
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Add the parent directory to the path so we can import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poker_core as pc
from poker.rl_interface import RLEnvironment

def create_poker_env(env_config):
    """
    Create a poker environment for RLlib.
    
    Args:
        env_config: Environment configuration
        
    Returns:
        A poker environment
    """
    return RLEnvironment(
        num_players=env_config.get("num_players", 3),
        buy_in=env_config.get("buy_in", 500),
        small_blind=env_config.get("small_blind", 10),
        big_blind=env_config.get("big_blind", 20)
    )

def create_model(obs_space, act_space, num_outputs, model_config, name):
    """
    Create a custom model for RLlib.
    
    Args:
        obs_space: Observation space
        act_space: Action space
        num_outputs: Number of output neurons
        model_config: Model configuration
        name: Model name
        
    Returns:
        A TensorFlow model
    """
    # Input layer
    input_layer = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
    
    # Hidden layers
    x = tf.keras.layers.Dense(256, activation="relu")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Output layer
    output_layer = tf.keras.layers.Dense(num_outputs, activation=None)(x)
    
    # Create model
    return tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

def train_rl_agent():
    """
    Train a reinforcement learning agent for poker.
    """
    # Register the environment with Ray
    register_env("poker_env", create_poker_env)
    
    # Create a PPO configuration
    config = (
        PPOConfig()
        .environment("poker_env", env_config={
            "num_players": 3,
            "buy_in": 500,
            "small_blind": 10,
            "big_blind": 20
        })
        .framework("tf2")
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "custom_model": None,  # Use create_model if you want a custom model
            },
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            entropy_coeff=0.01,
            clip_param=0.2,
            vf_clip_param=10.0,
        )
        .resources(num_gpus=0)  # Set to 1 if you have a GPU
        .rollouts(num_rollout_workers=4)
    )
    
    # Train the agent
    stop = {
        "training_iteration": 100,
        "timesteps_total": 1000000,
        "episode_reward_mean": 500,  # Stop when mean reward reaches 500
    }
    
    # Use tune.run to train the agent
    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir="./results",
        name="poker_ppo",
        verbose=1
    )
    
    # Get the best checkpoint
    best_checkpoint = results.get_best_checkpoint(
        results.get_best_trial("episode_reward_mean", "max"),
        "episode_reward_mean",
        "max"
    )
    
    print(f"Best checkpoint: {best_checkpoint}")
    return best_checkpoint

def evaluate_agent(checkpoint_path, num_episodes=100):
    """
    Evaluate a trained RL agent.
    
    Args:
        checkpoint_path: Path to the checkpoint
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Average reward
    """
    # Create environment
    env = RLEnvironment(num_players=3, buy_in=500, small_blind=10, big_blind=20)
    
    # Load the agent
    agent = PPOConfig().framework("tf2").build()
    agent.restore(checkpoint_path)
    
    # Evaluate
    total_reward = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    avg_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")
    
    return avg_reward

def simple_dqn_example():
    """
    A simplified DQN example without Ray/RLlib dependency.
    """
    # Create environment
    env = RLEnvironment(num_players=3, buy_in=500, small_blind=10, big_blind=20)
    
    # Create a simple Q-network
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')
    
    # Training parameters
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 64
    replay_buffer = []
    max_buffer_size = 10000
    
    # Training loop
    episodes = 1000
    max_steps = 1000
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Choose action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0])
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Store in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)
            
            # Train model
            if len(replay_buffer) >= batch_size:
                indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch = [replay_buffer[i] for i in indices]
                
                states = np.array([x[0] for x in batch])
                actions = np.array([x[1] for x in batch])
                rewards = np.array([x[2] for x in batch])
                next_states = np.array([x[3] for x in batch])
                dones = np.array([x[4] for x in batch])
                
                targets = model.predict(states, verbose=0)
                q_next = model.predict(next_states, verbose=0)
                
                for i in range(batch_size):
                    if dones[i]:
                        targets[i, actions[i]] = rewards[i]
                    else:
                        targets[i, actions[i]] = rewards[i] + gamma * np.max(q_next[i])
                
                model.train_on_batch(states, targets)
            
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon:.4f}")
    
    # Save the model
    model.save("poker_dqn_model.h5")
    print("Training complete. Model saved to poker_dqn_model.h5")

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Poker RL Examples")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "simple_dqn"], 
                        default="simple_dqn", help="Mode to run")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for evaluation")
    args = parser.parse_args()
    
    if args.mode == "train":
        checkpoint = train_rl_agent()
    elif args.mode == "evaluate":
        if not args.checkpoint:
            print("Please provide a checkpoint path for evaluation")
            return
        evaluate_agent(args.checkpoint)
    elif args.mode == "simple_dqn":
        simple_dqn_example()

if __name__ == "__main__":
    main()
