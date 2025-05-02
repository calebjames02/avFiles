import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Create environment
env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95)  # use "rgb_array" if you don't want real-time rendering
env = Monitor(env)  # Optional: to log metrics
env = DummyVecEnv([lambda: env])  # Wrap for stable-baselines3

# Create model
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./carracing_tensorboard/")

# Train model
model.learn(total_timesteps=100_000)

# Save model
model.save("ppo_carracing")

# To test the trained model
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
