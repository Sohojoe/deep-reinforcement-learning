import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

seed = 0
seed = 1
seed = 2
seed = 3
seed = 4

env = gym.make('CartPole-v1')
env.seed(seed)

observations_size = env.observation_space.shape[0]
action_size = env.action_space.n
print('State shape: ', observations_size)
print('Number of actions: ', action_size)

from dqn_agent import Agent

agent = Agent(state_size=observations_size, action_size=action_size, seed=seed)

# watch an untrained agent
# state = env.reset()
# for j in range(200):
#     action = agent.act(state)
#     env.render()
#     state, reward, done, _ = env.step(action)
#     if done:
#         break 
        
# env.close()


# 3. Train the Agent with DQN
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    high_score = 0
    has_new_high_score = False
    high_scores_window = deque(maxlen=10)
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        high_scores_window.append(score)  # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps))
            if has_new_high_score:
                print('\rNew High Score: {:.2f}'.format(high_score))
                has_new_high_score = False
        if np.mean(high_scores_window)>high_score:
            high_score = np.mean(high_scores_window)
            has_new_high_score = True
        if np.mean(scores_window)>=195.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            print('\rHigh Score: {:.2f}'.format(high_score))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# 4. Watch a Smart Agent!
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(5):
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
env.close()