import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import cv2
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tensorboard setup
run_name = f"PPO_CarRacing_{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")
save_dir = "debug_frames"

class DiscretizedCarRacing(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = [
            # Format: [steering, gas, brake]
            # https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/blob/master/CarRacingDQNAgent.py
            np.array([-1.0, 1.0, 0.2], dtype=np.float64),
            np.array([-1.0, 1.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.2], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.2], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.2], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 1.0, 0.2], dtype=np.float64),
            np.array([1.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.2], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        ]
        
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action_idx):
        return self.actions[action_idx]
        
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
env = DiscretizedCarRacing(env)

def preprocess_state(state, frame_id = time.time()):
    # https://hiddenbeginner.github.io/study-notes/contents/tutorials/2023-04-20_CartRacing-v2_DQN.html
    state = state[:84, 6:90] # Crop out background
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) # Make the image black and white and normalize it

    state = np.expand_dims(state, axis=0) # Add one extra channel at start (1, 84, 84)

#    state_save = np.transpose(state, (1, 2, 0)) # (84, 84, 1)
#    cv2.imwrite(os.path.join(save_dir, f"{frame_id}_original.png"), state_save) # For debugging

    state = state / 255.0

    return state

class ActorCritic(nn.Module):
    def __init__(self, action_dim, state_dim=1):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 64, kernel_size=8, stride=4) # 84 x 84 -> 20 x 20
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2) # 20 x 20 -> 9 x 9
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1) # 9 x 9 -> 7 x 7

        with torch.no_grad(): # Why torch.no_grad()?
            dummy_input = torch.zeros(1, state_dim, 84, 84)
            x = torch.relu(self.conv1(dummy_input))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            self.flatten_size = x.view(1, -1).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        
        logits = self.actor(x) # Create actor
        critic = self.critic(x) # Create critic

        return logits, critic

# Compute returns
def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    values = torch.stack(values) # Combine list of tensors into a single tensor for values
    
    last_value_tensor = torch.tensor([last_value], dtype=torch.float32, device=device)
    values = torch.cat([values, last_value_tensor]) # Combine value and last value tensors

    gae = 0 # Accumulator
    returns = []
    for step in reversed(range(len(rewards))): # Loop backward from last timestep to first
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae # Compute gae for current timestep
        returns.insert(0, gae + values[step])
        
    return torch.stack(returns)

model = ActorCritic(action_dim = env.action_space.n).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

#Hyperparameters
max_steps = 1_000_000
gamma = 0.99
ppo_epochs = 5
epsilon = 0.1
batch_size = 64
buffer_size = 2048

episode = 1
total_steps = 0
states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
while(total_steps < max_steps):
    state, _ = env.reset()
    done = False
    total_reward = 0
    episode_steps = 0

    while not done:
        state_proc = preprocess_state(state)
        state_t = torch.tensor(state_proc, dtype=torch.float32, device=device).unsqueeze(0).to(device)

        action_logits, value = model(state_t)
        dist = Categorical(logits = action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        states.append(state_t.squeeze(0))
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value.detach().squeeze())
        log_probs.append(log_prob)
        
        state = next_state
        total_reward += reward
        episode_steps += 1
        total_steps += 1
        
        # See if buffer is full
        if len(states) >= buffer_size:
            # Get last value
            if not done:
                with torch.no_grad():
                    state_proc = preprocess_state(state)
                    state_t = torch.tensor(state_proc, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension: [1, 3, 96, 96]
                    
                    _, last_value = model(state_t)
                    last_value = last_value.item()
            else:
                last_value = 0
                
            returns = compute_gae(rewards, dones, values, last_value) # Compute returns
            
            # Convert to tensors
            states_tensor = torch.stack(states)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
            old_log_probs_tensor = torch.stack(log_probs).detach()
            old_values_tensor = torch.stack(values).detach()
            
            advantages = returns - old_values_tensor # Compute advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize advantages
            
            # Create dataset for mini-batch updates
            dataset = torch.utils.data.TensorDataset(
                states_tensor, actions_tensor, old_log_probs_tensor, returns, advantages
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for _ in range(ppo_epochs): # PPO update loop
                for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                    # Forward pass
                    logits, values_pred = model(batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    
                    # Ratio for PPO
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    # PPO losses
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * batch_advantages
                    
                    # Policy loss
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = 0.5 * ((values_pred.squeeze() - batch_returns) ** 2).mean()
                    
                    # Entropy bonus
                    entropy = dist.entropy().mean()
                    
                    # Total loss
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    
                    # Optimization step
                    optimizer.zero_grad()
                    loss.backward()
            
            # Reset experience buffer
            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
            
            # Log training metrics
            writer.add_scalar("Loss/Policy", policy_loss.item(), episode)
            writer.add_scalar("Loss/Value", value_loss.item(), episode)
            writer.add_scalar("Action/Entropy", entropy.item(), episode)
            
            writer.flush()

    # Log episode results
    writer.add_scalar("Reward/Total", total_reward, episode)
    writer.add_scalar("Episode/Length", episode_steps, episode)
    writer.flush()
    
    print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Length = {episode_steps}, Steps = {total_steps}")
    
    episode += 1
