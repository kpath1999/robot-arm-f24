import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from scripts.picknplace import PickAndPlaceEnv
# Instantiate the environment
env = PickAndPlaceEnv()

# Optional: Check the environment
check_env(env)

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_pick_and_place_tensorboard/")

# Train the agent
model.learn(total_timesteps=200000)

# Save the trained model
model.save("ppo_pick_and_place")

# Close the environment
env.close()