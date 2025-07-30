import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

from gym.wrappers import GrayScaleObservation, ResizeObservation
import os
import numpy as np

from gym import RewardWrapper

class CustomRewardWrapper(RewardWrapper):
    def reward(self, reward):
                return max(reward, 0)




def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env, keep_dim=True)
    env = Monitor(env)
    env = CustomRewardWrapper(env)  
    return env

# Wrap environment
env = DummyVecEnv([create_env])
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# Save path
save_path = os.path.join("train", "ppo_mario")
os.makedirs(save_path, exist_ok=True)

# Define PPO model
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_mario_logs/", ent_coef=0.05)

# Train
model.learn(total_timesteps=500000)
model.save(os.path.join(save_path, "mario_model"))

print("Training complete and model saved.")


