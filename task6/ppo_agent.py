import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class DiscretizedCarRacing(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = [
            np.array([0.0, 0.0, 0.0]),   # No-op
            np.array([0.0, 1.0, 0.0]),   # Full throttle
            np.array([0.0, 0.0, 0.8]),   # Brake
            np.array([-1.0, 0.2, 0.0]),  # Full steer left + throttle
            np.array([1.0, 0.2, 0.0]),   # Full steer right + throttle
            np.array([-0.5, 0.4, 0.0]),  # Medium steer left + throttle
            np.array([0.5, 0.4, 0.0]),   # Medium steer right + throttle
        ]

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]

if torch.cuda.is_available(): # move tensors to cuda
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')

# Create the environment
env = gym.make("CarRacing-v3", render_mode="human")
env = DiscretizedCarRacing(env)
log_dir = f"runs/experiment_{int(time.time())}"
writer = SummaryWriter(log_dir=log_dir)
obs_shape = env.observation_space.shape  # (96, 96, 3) for car racing env
action_dim = env.action_space.n

class Policy(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # Converts image from 96 x 96 to 23 x 23
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Converts image from 23 x 23 to about 10 x 10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Converts image from 10 x 10 to about 8 x 8
            nn.ReLU()
        )

        with torch.no_grad(): # Compute CNN output size
            dummy_input = torch.zeros(1, 3, 96, 96)
            conv_out = self.cnn(dummy_input)
            conv_out_size = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential( # Set up CNN
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(128, num_actions) # actor
        self.value_head = nn.Linear(128, 1) # critic

    def forward(self, x): # Breaks down image for CNN
        x = x / 255.0  # Normalize image
        x = self.cnn(x)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


policy = Policy(num_actions = env.action_space.n).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

#Hyperparameters
gamma = 0.99
ppo_epochs = 5
epsilon = 0.1
batch_size = 64
buffer_size = 2048

# Compute returns
def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device) # Convert rewards to tensor
    dones = torch.tensor(dones, dtype=torch.float32, device=device) # Convert dones to tensor
    values = torch.stack(values) # Combine list of tensors into a single tensor for values
    
    last_value_tensor = torch.tensor([last_value], dtype=torch.float32, device=device) # Convert last value to tensor
    values = torch.cat([values, last_value_tensor]) # Combine value and last value tensors

    gae = 0 # Accumulator
    returns = []
    for step in reversed(range(len(rewards))): # Loop backward from last timestep to first
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae # Compute gae for current timestep
        returns.insert(0, gae + values[step])
        
    return torch.stack(returns)

# Training parameters
max_steps = 1_000_000  # Maximum training steps
exploration_rate = 0.1  # Initial exploration rate

episode = 1
total_steps = 0
states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

while total_steps < max_steps:
    state, _ = env.reset()
    done = False
    total_reward = 0
    episode_steps = 0
    
    # Decay exploration rate
    exploration_rate = max(0.01, exploration_rate * 0.995)  # Minimum 1% exploration

    while not done:
        # Preprocess state
        state_t = torch.tensor(np.transpose(state, (2, 0, 1)), dtype=torch.float32, device=device).unsqueeze(0)

        # Forward pass
        logits, value = policy(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        
        # Epsilon-greedy action selection for exploration
        if np.random.random() < exploration_rate:
            action = env.action_space.sample()
            # Get log probability for the sampled action
            action_tensor = torch.tensor([action], device=device)
            log_prob = dist.log_prob(action_tensor)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()

        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store experience
        states.append(state_t.squeeze(0))
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value.detach().squeeze())

        state = next_state
        total_reward += reward
        episode_steps += 1
        total_steps += 1

        # See if buffer is full
        if len(states) >= buffer_size or done:
            # Get last value
            if not done:
                with torch.no_grad():
                    _, last_value = policy(torch.tensor(np.transpose(state, (2, 0, 1)), 
                                         dtype=torch.float32, device=device).unsqueeze(0))
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
                    logits, values_pred = policy(batch_states)
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
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                    optimizer.step()
            
            # Reset experience buffer
            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
            
            # Log training metrics
            writer.add_scalar("Loss/Policy", policy_loss.item(), total_steps)
            writer.add_scalar("Loss/Value", value_loss.item(), total_steps)
            writer.add_scalar("Action/Entropy", entropy.item(), total_steps)
            writer.add_scalar("Training/Exploration_rate", exploration_rate, total_steps)

    # Log episode results
    writer.add_scalar("Reward/Total", total_reward, episode)
    writer.add_scalar("Episode/Length", episode_steps, episode)
    
    print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Length = {episode_steps}, Steps = {total_steps}, Exploration = {exploration_rate:.3f}")

    episode += 1

writer.close()
env.close()
