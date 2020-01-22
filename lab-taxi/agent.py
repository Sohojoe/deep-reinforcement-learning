import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.epsilon = None
        self.nA = nA
        # self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.Q = defaultdict(lambda: np.full(self.nA, 01.30))

        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = .2
        self.i_episode = 1
        # self.alg = 'q'
        # self.alg = 'saras'
        self.alg = 'expected_saras'
        
        self.episode_step = 0
        self.next_action = None
    
    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)
        return np.random.choice(ties)

    def e_greedy_sample(self, state):
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None
            return action
        if np.random.random() >= self.epsilon:
            action = self.argmax(self.Q[state])
        else:
            action = np.random.choice(self.nA)
        return action

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # action = np.random.choice(self.nA)
        action = self.e_greedy_sample(state) 
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done and (self.i_episode % 2500 ==0):
            print("", self.epsilon, self.alpha)

        value = self.Q[state][action]
        if self.alg == 'saras':
            next_action = self.e_greedy_sample(state)
            next_value = self.Q[next_state][next_action]
        elif self.alg == 'q':
            next_value = self.argmax(self.Q[next_state])
        elif self.alg == 'expected_saras':
            next_value = np.ones(self.nA) * self.epsilon/self.nA
            next_value[self.argmax(self.Q[next_state])] += 1.-self.epsilon
            next_value = np.dot(self.Q[next_state], next_value)
        # next_value = 0. if done else next_value
        target_value = reward + self.gamma * next_value
        self.Q[state][action] = value + (self.alpha*(target_value - value))
        self.episode_step += 1
        if done:
            self.episode_step = 0
            self.next_action = None
            self.i_episode += 1
            self.epsilon /= 2.
#             self.epsilon = max(self.epsilon, 0.03)