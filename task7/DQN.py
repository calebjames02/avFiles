import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2
import time
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Discretize actions ---
class DiscretizedCarRacing(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = [
            np.array([-1.0, 1.0, 0.0]),   # Left 
            np.array([0.0, 1.0, 0.0]),    # Straight 
            np.array([1.0, 1.0, 0.0]),    # Right 
            np.array([0.0, 0.0, 0.8]),    # Brake
            np.array([0.0, 0.0, 0.0]),    # Do nothing
        ]
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action_idx):
        return self.actions[action_idx]

def preprocess(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return np.expand_dims(resized, axis=0) / 255.0

class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


def train_dqn():
    env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
    env = DiscretizedCarRacing(env)
    action_dim = env.action_space.n

    q_net = DQN(action_dim).to(device)
    target_net = DQN(action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    buffer = ReplayBuffer()

    run_name = f"dqn_carracing_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 64
    target_update_freq = 1000

    total_steps = 0

    for episode in range(1000):
        state, _ = env.reset()
        state = preprocess(state)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        done = False
        episode_reward = 0
        episode_loss = []

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    action = q_values.argmax().item()

            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess(next_state_raw)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

            buffer.push((state.cpu().numpy(), action, reward, next_state_tensor.cpu().numpy(), done))

            state = next_state_tensor
            episode_reward += reward
            total_steps += 1

            # Training
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32).squeeze(1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).squeeze(1).to(device)
                actions = torch.tensor(actions, dtype=torch.long).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q = target_net(next_states).max(1)[0]
                    target = rewards + gamma * target_q * (1 - dones)

                loss = nn.MSELoss()(q_values, target)
                episode_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # TensorBoard logging
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        if episode_loss:
            writer.add_scalar("Loss/Episode", np.mean(episode_loss), episode)
        writer.flush()

        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")

    writer.close()
    env.close()

if __name__ == "__main__":
    train_dqn()
