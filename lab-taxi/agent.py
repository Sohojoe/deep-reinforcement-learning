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
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        # self.gamma = 1.
        self.gamma = 0.9
        self.epsilon = 0.005
        self.alpha = 0.2
        self.i_episode = 1
    
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
        random_action = np.random.choice(self.nA)
        greedy_action = self.argmax(self.Q[state]) if state in self.Q else random_action
        action = greedy_action if np.random.random() >= self.epsilon \
            else random_action
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
        # self.Q[state][action] += 1
        # s1, r1, done, info = env.step(a)
        # a1 = e_greedy_sample(Q, s1, epsilon, env.action_space)
        
        self.epsilon = 1.0 / self.i_episode

        value = self.Q[state][action]
        scale = np.ones(self.nA) * self.epsilon/self.nA
        scale[self.argmax(self.Q[next_state])] += 1.-self.epsilon
        scale = np.dot(self.Q[next_state], scale)
        target_value = reward + self.gamma * scale if not done else 0
#             target_value = r1 + (gamma * scale)
        self.Q[state][action] = value + (self.alpha*(target_value - value))
        if done:
            self.i_episode += 1        