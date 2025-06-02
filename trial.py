import gym
import numpy as np




env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()  # use unpacking if you're on Gym v0.26+

state = env.encode(3,1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.unwrapped.s = state #in Gym v0.26+, env is wrapped in multiple layers (like OrderEnforcing, EnvChecker, etc.). These wrappers don’t expose env.s directly.
output = env.render()
print(output)

print(env.P[328])
#print(env.action_space)
#print(env.observation_space)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

"""Training the agent"""

import random
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    state, _ = env.reset()
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Define `done` yourself
 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
print(q_table[328])

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties, total_rewards = 0, 0, 0
episodes = 10000

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    state, _ = env.reset()
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Combine to get the old-style `done`
        state = next_state


        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs
    total_rewards += reward

print(f"Results after {episodes} episodes:")
print(f"Total timesteps per episode: {total_epochs}")
print(f"Total penalties per episode: {total_penalties}")
print(f"Total rewards per episode: {total_rewards}")
print("Average timesteps per episode:", total_epochs / episodes)
print("Average penalties per episode:", total_penalties / episodes)
print("Average rewards per episode:", total_rewards / episodes)
