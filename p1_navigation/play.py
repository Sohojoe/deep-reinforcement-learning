from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import random
from dqn_agent import Agent

env = UnityEnvironment(file_name="Banana.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
action_size = brain.vector_action_space_size
state_size = len(state)

agent = Agent(state_size=state_size, action_size=action_size, seed=0)


# 4. Watch a Smart Agent!
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

high_score = 0
for i in range(5):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    for j in range(200):
        action = agent.act(state)
        # env.render()
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        state = next_state
        score += reward
        if score > high_score:
            high_score = score
        if reward != 0:
            print('got reward', reward, 'score:', score)
        if done:
            break 
    print('episode', i, 'score:', score)
print('high_score', high_score)
            
env.close()