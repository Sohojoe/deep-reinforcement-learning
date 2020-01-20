import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values


env = gym.make('CliffWalking-v0')

print(env.action_space)
print(env.observation_space)


# define the optimal state-value function
V_opt = np.zeros((4,12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

# plot_values(V_opt)
def argmax(q_values):
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)
def e_greedy_sample(Q, state, pi_epsilon, action_space):
    random_action = action_space.sample()
    greedy_action = argmax(Q[state]) if state in Q else random_action
    action = greedy_action if np.random.random() >= pi_epsilon \
        else random_action
    return action

def sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        ## TODO: complete the function
        epsilon = 1.0 / i_episode
        s = env.reset()
        while True:
            a = e_greedy_sample(Q, s, epsilon, env.action_space)
            s1, r1, done, info = env.step(a)
            a1 = e_greedy_sample(Q, s1, epsilon, env.action_space)
            value = Q[s][a]
#             target_value = r1 + (gamma * Q[s1][a1]) if not done else 0
            target_value = r1 + gamma * Q[s1][a1] if not done else 0
            Q[s][a] = value + alpha*(target_value - value)
            # Q[s][a] = Q[s][a] + alpha * (r1 + gamma * Q[s1][a1] - Q[s][a])
            # Q[s][a] = Q[s][a] + (alpha * (r1 + ((gamma * Q[s1][a1]) - Q[s][a])))
            s = s1
            if done: 
                break
    return Q

run_part_one = False
# run_part_one = True
if run_part_one:
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsa = sarsa(env, 5000, .01)
    # Q_sarsa = sarsa(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_sarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsa)

    # plot the estimated optimal state-value function
    V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)

# Part 2: TD Control: Q-learning 
def e_greedy_sample(Q, state, pi_epsilon, action_space):
    random_action = action_space.sample()
    greedy_action = argmax(Q[state]) if state in Q else random_action
    action = greedy_action if np.random.random() >= pi_epsilon \
        else random_action
    return action


def q_learning(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        ## TODO: complete the function
        epsilon = 1.0 / i_episode
        s = env.reset()
        while True:
            a = e_greedy_sample(Q, s, epsilon, env.action_space)
            s1, r1, done, info = env.step(a)
            # a1 = e_greedy_sample(Q, s1, epsilon, env.action_space)
            value = Q[s][a]
            target_value = r1 + gamma * max(Q[s1]) if not done else 0
            Q[s][a] = value + alpha*(target_value - value)
            s = s1
            if done: 
                break
    return Q

run_part_two = False
# run_part_two = True
if run_part_two:
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsamax = q_learning(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
    check_test.run_check('td_control_check', policy_sarsamax)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsamax)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])


# Part 3: TD Control: Expected Sarsa
def e_greedy_sample(Q, state, pi_epsilon, action_space):
    random_action = action_space.sample()
    greedy_action = argmax(Q[state]) if state in Q else random_action
    action = greedy_action if np.random.random() >= pi_epsilon \
        else random_action
    return action

def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        ## TODO: complete the function
#         epsilon = 1.0 / i_episode
        epsilon = 0.005
        s = env.reset()
        while True:
            a = e_greedy_sample(Q, s, epsilon, env.action_space)
            s1, r1, done, info = env.step(a)
            # a1 = e_greedy_sample(Q, s1, epsilon, env.action_space)
            value = Q[s][a]
            scale = np.ones(env.nA) * epsilon/env.nA
            scale[argmax(Q[s1])] += 1.-epsilon
            scale = np.dot(Q[s1], scale)
            target_value = r1 + gamma * scale if not done else 0
#             target_value = r1 + (gamma * scale)
            Q[s][a] = value + (alpha*(target_value - value))
            s = s1
            if done: 
                break
    return Q

run_part_three = False
run_part_three = True
if run_part_three:
    # obtain the estimated optimal policy and corresponding action-value function
    Q_expsarsa = expected_sarsa(env, 10000, 1)

    # print the estimated optimal policy
    policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_expsarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_expsarsa)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])