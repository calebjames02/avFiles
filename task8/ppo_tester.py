import os
import time
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def preprocess_state(state, frame_id = time.time()):
    # https://hiddenbeginner.github.io/study-notes/contents/tutorials/2023-04-20_CartRacing-v2_DQN.html
#    state = state[:84, 6:90] # Crop out background
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) # Make the image black and white and normalize it

    state = np.expand_dims(state, axis=0) # Add one extra channel at start (1, 84, 84)

#    state_save = np.transpose(state, (1, 2, 0)) # (84, 84, 1)
#    cv2.imwrite(os.path.join(save_dir, f"{frame_id}_original.png"), state_save) # For debugging

    state = state / 255.0

    return torch.tensor(state, dtype=torch.float32, device=device)

class ActorCritic(nn.Module):
    def __init__(self, action_dim, state_dim=1):
        super(ActorCritic, self).__init__()
        # https://web.stanford.edu/class/aa228/reports/2019/final9.pdf
        self.conv1 = nn.Conv2d(state_dim, 64, kernel_size=8, stride=4) # 84 x 84 -> 20 x 20
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2) # 20 x 20 -> 9 x 9
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1) # 9 x 9 -> 7 x 7

        with torch.no_grad():
            # http://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
            dummy_input = torch.zeros(1, state_dim, 96, 96)
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

class RolloutBuffer():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def add(self, state, action, reward, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def get_tensors(self, returns, advantages):
        states_tensor = torch.stack(self.states)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=device)
        old_log_probs_tensor = torch.stack(self.log_probs)
       
        return states_tensor, actions_tensor, old_log_probs_tensor, returns, advantages

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        values = torch.stack(self.values) # Combine list of tensors into a single tensor for values

        last_value_tensor = torch.tensor([last_value], dtype=torch.float32, device=device)
        values_extended = torch.cat([values, last_value_tensor]) # Combine value and last value tensors

        gae = 0 # Accumulator
        returns = []
        for step in reversed(range(len(rewards))): # Loop backward from last timestep to first
            delta = rewards[step] + gamma * values_extended[step + 1] * (1 - dones[step]) - values_extended[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae # Compute gae for current timestep
            returns.insert(0, gae + values_extended[step])
       
        returns = torch.stack(returns)
        advantages = returns - values
       
        return returns, advantages

class PPO():
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        n_steps=2000,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        reward_scale = 1,
        ):

        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.reward_scale = reward_scale
       
        # Initialize network directly with state_dim=1 for grayscale images
        self.policy = ActorCritic(action_dim=env.action_space.n, state_dim=1).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
       
        # Initialize buffer
        self.rollout_buffer = RolloutBuffer()
        
    def scale_reward(self, reward):
        if reward > -10:
            reward *= self.reward_scale
            
        return reward
       
    def collect_rollouts(self):
        episode_reward = 0
        episode_steps = 1
       
        while(episode_steps <= self.n_steps):
            state, _ = env.reset()
            done = False
       
            while not done:
                state_proc = preprocess_state(state).unsqueeze(0)

                with torch.no_grad():
                    action_logits, value = self.policy(state_proc)

                dist = Categorical(logits = action_logits)
               
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.item()
               
                next_state, reward, terminated, truncated, _ = env.step(action)
                reward = self.scale_reward(reward)
               
                done = terminated or truncated
               
                self.rollout_buffer.add(state_proc.squeeze(0), action, reward, done, value.detach().squeeze(), log_prob)
               
                state = next_state
                episode_reward += reward
                episode_steps += 1
           
        if not done:
            with torch.no_grad():
                state_proc = preprocess_state(state).unsqueeze(0)
                   
                _, last_value = self.policy(state_proc)
                last_value = last_value.item()
        else:
            last_value = 0

        returns, advantages = self.rollout_buffer.compute_returns_and_advantages(last_value)
       
        return returns, advantages, episode_reward
       
    def update_policy(self, returns, advantages):
        states_tensor, actions_tensor, old_log_probs_tensor, returns, advantages = self.rollout_buffer.get_tensors(returns, advantages)
       
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset for update loop
        # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor, old_log_probs_tensor.detach(), returns.detach(), advantages.detach())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.n_epochs): # PPO update loop
            for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                # Forward pass
                logits, values_pred = self.policy(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                   
                # Ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                   
                # PPO losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                   
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                   
                # Value loss
                value_loss = 0.5 * ((values_pred.squeeze() - batch_returns) ** 2).mean()
                   
                # Entropy bonus
                entropy = dist.entropy().mean()
                   
                # Total loss
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                 
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
           
        # Reset experience buffer
        self.rollout_buffer.clear()
           
        return policy_loss.item(), value_loss.item(), entropy.item()
       
    def learn(self, total_timesteps = 1_000_000):
        timesteps_so_far = 0
        iteration = 0
       
        while timesteps_so_far < total_timesteps:
            # Collect rollouts
            returns, advantages, episode_reward = self.collect_rollouts()
           
            # Update policy
            policy_loss, value_loss, entropy_loss = self.update_policy(returns, advantages)
           
            # Update timesteps
            timesteps_so_far += self.n_steps
            iteration += 1
           
            # Log to tensorboard
            writer.add_scalar("losses/policy_loss", policy_loss, iteration)
            writer.add_scalar("losses/value_loss", value_loss, iteration)
            writer.add_scalar("losses/entropy_loss", entropy_loss, iteration)
            writer.add_scalar("charts/episode_reward", episode_reward, iteration)
           
            # Console logging
            print(f"Iteration {iteration}, Steps: {timesteps_so_far}")
            print(f"Mean reward: {episode_reward:.2f}")
            print(f"Losses: Policy={policy_loss:.4f}, Value={value_loss:.4f}, Entropy={entropy_loss:.4f}")
            print("-" * 50)
               
            writer.flush()
            
class PPO_Tester():
    def __init__(self):
        self.total_timesteps = 10_000_000
        self.episode_steps = 2_000
    
        self.ppo1 = PPO(
            env=env,
            learning_rate=3e-4,
            n_steps=10000,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            reward_scale = 1,)
        self.ppo2 = PPO(
            env=env,
            learning_rate=3e-4,
            n_steps=10000,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            reward_scale = 2,)
        self.ppo3 = PPO(
            env=env,
            learning_rate=3e-4,
            n_steps=10000,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            reward_scale = 6,)
        self.ppo4 = PPO(
            env=env,
            learning_rate=3e-4,
            n_steps=10000,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            reward_scale = 10,)
            
    def learn(self):
        current_timesteps = 0
        iteration = 0
        while current_timesteps < self.total_timesteps:
            # Collect rollouts
            returns1, advantages1, episode_reward1 = self.ppo1.collect_rollouts()
            returns2, advantages2, episode_reward2 = self.ppo2.collect_rollouts()
            returns3, advantages3, episode_reward3 = self.ppo3.collect_rollouts()
            returns4, advantages4, episode_reward4 = self.ppo4.collect_rollouts()
           
            # Update policy
            policy_loss1, value_loss1, entropy_loss1 = self.ppo1.update_policy(returns1, advantages1)
            policy_loss2, value_loss2, entropy_loss2 = self.ppo2.update_policy(returns2, advantages2)
            policy_loss3, value_loss3, entropy_loss3 = self.ppo3.update_policy(returns3, advantages3)
            policy_loss4, value_loss4, entropy_loss4 = self.ppo4.update_policy(returns4, advantages4)

            current_timesteps += self.episode_steps
            iteration += 1
            
            episode_reward1 /= self.ppo1.reward_scale
            episode_reward2 /= self.ppo2.reward_scale
            episode_reward3 /= self.ppo3.reward_scale
            episode_reward4 /= self.ppo4.reward_scale

            # Log to tensorboard
            writer.add_scalars("policy_loss", {
                "group_1": policy_loss1,
                "group_2": policy_loss2,
                "group_3": policy_loss3,
                "group_4": policy_loss4,
            }, iteration)

            writer.add_scalars("value_loss", {
                "group_1": value_loss1,
                "group_2": value_loss2,
                "group_3": value_loss3,
                "group_4": value_loss4,
            }, iteration)

            writer.add_scalars("entropy_loss", {
                "group_1": entropy_loss1,
                "group_2": entropy_loss2,
                "group_3": entropy_loss3,
                "group_4": entropy_loss4,
            }, iteration)

            writer.add_scalars("episode_reward", {
                "group_1": episode_reward1,
                "group_2": episode_reward2,
                "group_3": episode_reward3,
                "group_4": episode_reward4,
            }, iteration)

            # Console logging
            print(f"Iteration {iteration}, Steps: {current_timesteps}")
   
# Tensorboard setup
run_name = f"PPO_CarRacing_{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")
save_dir = "debug_frames"
   
# Environment setup
env = gym.make("CarRacing-v3", render_mode=None, continuous=True)
env = DiscretizedCarRacing(env)

ppo_tester = PPO_Tester()
ppo_tester.learn()
