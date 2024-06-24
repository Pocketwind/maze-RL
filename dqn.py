import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from game_v2 import Game_v2

# DQN parameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory_size = 1000000
learning_rate = 0.001
target_update_frequency = 1000

# Create the game environment
env = Game_v2()

# Create the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

q_network = QNetwork(env.observation_space.n, env.action_space.n)
q_network_target = QNetwork(env.observation_space.n, env.action_space.n)
q_network_target.load_state_dict(q_network.state_dict())
q_network_target.eval()

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Create the replay memory
memory = deque(maxlen=memory_size)

# Training loop
episode_rewards = []
for episode in range(1, 10001):
    state = env.reset()
    print(state)
    done = False
    episode_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_network(torch.FloatTensor([state])).detach().numpy()[0]
            action = np.argmax(q_values)

        next_state, reward, done, _ = env.step(action)
        next_state = next_state
        episode_reward += reward

        # Add the transition to the replay memory
        memory.append((state, action, reward, next_state, done))

        # Update the Q-network
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = q_network_target(next_states).max(1)[0].detach()
            targets = rewards + gamma * (1 - dones) * next_q_values

            loss = F.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the target network
            if episode % target_update_frequency == 0:
                q_network_target.load_state_dict(q_network.state_dict())

        state = next_state

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print episode information
    episode_rewards.append(episode_reward)
    if episode % 100 == 0:
        print(f"Episode {episode}: epsilon = {epsilon:.3f}, average reward = {np.mean(episode_rewards[-100:]):.3f}")

# Test loop
state = env.reset().flatten()
done = False

while not done:
    q_values = q_network(torch.FloatTensor([state])).detach().numpy()[0]
    action = np.argmax(q_values)
    next_state, reward, done, _ = env.step(action)
    next_state = next_state
    state = next_state

    env.render()

print("Game over!")