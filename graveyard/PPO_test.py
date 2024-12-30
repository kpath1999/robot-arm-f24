import gym
from stable_baselines3 import PPO
from scripts.picknplace import PickAndPlaceEnv
# Instantiate the environment
env = PickAndPlaceEnv()

# Load the trained model
model = PPO.load("ppo_pick_and_place")

# Run evaluation episodes
num_episodes = 5
for episode in range(num_episodes):
    obs,_ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(total_reward, done)
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

# Close the environment
env.close()