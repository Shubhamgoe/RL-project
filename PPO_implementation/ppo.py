import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random

# Hyperparameters
learning_rate = 3e-4
gamma = 0.99  # Discount factor for rewards
clip_epsilon = 0.2  # PPO clipping parameter
epochs = 4  # Number of epochs to run PPO
batch_size = 64
buffer_size = 2048  # Replay buffer size
update_interval = 2000  # After how many steps we update the model

# Define the Policy and Value networks
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # Output a probability distribution over actions
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a value for the state
        )
        
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

# Define PPO Agent
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.actor_critic = ActorCritic(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.buffer = deque(maxlen=buffer_size)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs, _ = self.actor_critic(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, masks, next_value):
        returns = []
        G = next_value
        for reward, mask in zip(rewards[::-1], masks[::-1]):
            G = reward + gamma * G * mask
            returns.append(G)
        returns.reverse()
        return returns

    def update(self):
        states = []
        actions = []
        old_log_probs = []
        returns = []
        values = []
        rewards = []
        masks = []

        for experience in self.buffer:
            states.append(experience[0])
            actions.append(experience[1])
            old_log_probs.append(experience[2])
            rewards.append(experience[3])
            masks.append(experience[4])
            values.append(experience[5])

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        returns = torch.tensor(self.compute_returns(rewards, masks, values[-1]), dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        # Advantage estimation
        advantages = returns - values

        for _ in range(epochs):
            # Shuffle indices
            indices = np.random.permutation(len(states))
            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Compute the current action probabilities and value estimates
                action_probs, value = self.actor_critic(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)

                # Calculate the ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                # Surrogate loss function
                surrogate_loss = ratio * batch_advantages
                surrogate_loss_clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surrogate_loss, surrogate_loss_clipped).mean()

                # Value function loss (mean squared error)
                value_loss = (batch_returns - value).pow(2).mean()

                # Total loss (actor loss + value loss)
                loss = actor_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()

                # Perform backpropagation and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, num_episodes):
        episode_rewards = []
        episode_lengths = []
        for episode in range(num_episodes):
            state = self.env.reset()  # Ensure that we only get the state, not the full info
            state = state[0] if isinstance(state, tuple) else state  # For environments like CartPole
            done = False
            total_reward = 0
            total_length = 0
            rewards = []
            masks = []
            actions = []
            log_probs = []
            values = []

            while not done:
                action, log_prob = self.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)  # Include the info from reset
                next_state = next_state if isinstance(next_state, np.ndarray) else next_state[0]

                total_reward += reward
                total_length += 1

                # Store experience in buffer
                _, value = self.actor_critic(torch.tensor(state, dtype=torch.float32))
                self.buffer.append((state, action, log_prob, reward, 1 - done, value.item()))

                state = next_state

                if len(self.buffer) >= update_interval:
                    self.update()

            episode_rewards.append(total_reward)
            episode_lengths.append(total_length)

            # Print episode stats
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward}, Length: {total_length}")

        return episode_rewards, episode_lengths


if __name__ == "__main__":
    # Create the environment
    env = gym.make('CartPole-v1',render_mode="human")

    # Initialize PPO agent
    agent = PPOAgent(env)

    # Train the agent
    num_episodes = 1000
    agent.train(num_episodes)
    