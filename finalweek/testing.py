import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from gym.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import time
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

# Create env with all wrappers
env = DummyVecEnv([create_env])
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)  # if used in training

# Pass actual spaces to override the stored ones
model = PPO.load(
    "train/ppo_mario/mario_model",
    env=env,
    custom_objects={
        "observation_space": env.observation_space,
        "action_space": env.action_space
    }
)

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)

env.close()

