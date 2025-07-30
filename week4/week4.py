import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)

state = env.reset()
done = False

while not done:
    env.render()  # Shows the game screen
    action = env.action_space.sample()  # Random action
    state, reward, done, info = env.step(action)
    #print(info)
    print(state.shape)
env.close()