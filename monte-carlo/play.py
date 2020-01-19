import sys
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy



# frepisodeed = [((16, 9, False), 1, 0), ((21, 9, False), 0, 1.0)]
# for (state, action, reward) in episode:
#     print ('state', state, 'action', action, 'reward', reward)

# # episode = generate_episode()
# episode_len = len(episode)
# _, _, final_reward = episode[episode_len-1]
# for i in range(episode_len):

env = gym.make('Blackjack-v0')

print(env.observation_space)
print(env.action_space)

for i_episode in range(3):
    state = env.reset()
    while True:
        print(state)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print('End game! Reward: ', reward)
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break

def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

for i in range(3):
    # print(generate_episode_from_limit_stochastic(env))
    episode = generate_episode_from_limit_stochastic(env)
    for t, (s,a,r) in enumerate(episode):
        print ('t', t, 's',s, 'a', a, 'r', r)
    print('----')

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        ## TODO: complete the function
        episode = generate_episode(env)
        first_visit = []
        for t, (s,a,r) in enumerate(episode):
            if (s,a) not in first_visit:
                first_visit.append((s,a))
                G = 0
                for f in range(t,len(episode)):
                    _,_,next_reward = episode[f]
                    G += next_reward*pow(gamma, f-t)
                N[s][a] += 1.
                returns_sum[s][a] += G
                # Q[s][a] = returns_sum[s][a] / N[s][a]
    for s,_ in N.items():
        Q[s] = returns_sum[s] / N[s]
    return Q


# # obtain the action-value function
# # Q = mc_prediction_q(env, 5000, generate_episode_from_limit_stochastic)
# Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic)

# # obtain the corresponding state-value function
# V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
#          for k, v in Q.items())

# # plot the state-value function
# plot_blackjack_values(V_to_plot)


#----------- ---
# roll out episode, using Q and pi with epsilon
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

def generate_episode_using_pi(bj_env, pi, pi_epsilon):
    episode = []
    state = bj_env.reset()
    while True:
        random_action = env.action_space.sample()
        greedy_action = pi[state] if state in pi else random_action
        action = greedy_action if np.random.random() >= pi_epsilon \
            else random_action
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def score_policy(policy, assessment_games = 100000):
    won = 0
    lost = 0
    drawn = 0
    for i in range(assessment_games):
        episode = generate_episode_using_pi(env, policy, 0)
        _, _, rewards = zip(*episode)
        score = sum(rewards)
        if score > 0: won +=1
        elif score <0: lost +=1
        else: drawn +=1
    win_pc  = won*1./assessment_games * 100.
    lost_pc = lost*1./assessment_games * 100.
    draw_pc = drawn*1./assessment_games * 100.
    win_rate = win_pc / (win_pc+lost_pc) * 100.
    print ("Win rate =",  round(win_rate,1), \
        "%. Won", round(win_pc,1), "%. Lost", round(lost_pc,1), "%. Drawn", round(draw_pc,1), "%.")

def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(lambda: np.zeros(nA))
    policy = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # if i_episode > 0 and i_episode % 50000 == 0:
        #     score_policy(policy)
            # score_policy(policy)
            # score_policy(policy)
        
        ## TODO: complete the function
        # roll out episode, using Q and pi with epsilon
        def generate_episode_using_pi(bj_env, pi, pi_epsilon):
            episode = []
            state = bj_env.reset()
            while True:
                random_action = env.action_space.sample()
                greedy_action = pi[state] if state in pi else random_action
                action = greedy_action if np.random.random() >= pi_epsilon \
                    else random_action
                next_state, reward, done, info = bj_env.step(action)
                episode.append((state, action, reward))
                state = next_state
                if done:
                    break
            return episode

        # sets the whole policy, but is slow
        # for s, v in Q.items():
        #     policy[s] = np.argmax(v)
        epsilon = max(epsilon*eps_decay, eps_min)
        episode = generate_episode_using_pi(env, policy, epsilon)

        # update Q table
        first_visit = []
        for t, (s,a,r) in enumerate(episode):
            if (s,a) not in first_visit:
                first_visit.append((s,a))
                G = 0
                for f in range(t,len(episode)):
                    _,_,next_reward = episode[f]
                    G += next_reward*pow(gamma, f-t)
                # Incremental Mean
                # N[s][a] += 1.
                # Q[s][a] = Q[s][a] + 1./N[s][a] * (G-Q[s][a])
                # Constant-alpha
                Q[s][a] = (1-alpha) * Q[s][a] + alpha * G
                # just update the policy on a change
                policy[s] = argmax(Q[s])  

    return policy, Q


policy, Q = mc_control(env, 30000000, .0003, 1.)
# policy, Q = mc_control(env, 1000000, .003, 1.)
# policy, Q = mc_control(env, 1000000, .1, 1.)
# policy, Q = mc_control(env, 500000, .2, .95)
# policy, Q = mc_control(env, 500, .2, .95)

print ("")
# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

score_policy(policy)

# # plot the state-value function
plot_blackjack_values(V)

# # plot the policy
plot_policy(policy)

print ('----- done -----')
