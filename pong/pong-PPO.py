#!/usr/bin/env python
# coding: utf-8

# # Welcome!
# Below, we will learn to implement and train a policy to play atari-pong, using only the pixels as input. We will use convolutional neural nets, multiprocessing, and pytorch to implement and train our policy. Let's get started!
# 
# (I strongly recommend you to try this notebook on the Udacity workspace first before running it locally on your desktop/laptop, as performance might suffer in different environments)
# 
# # In[1]:


# # Install package for displaying animation
# get_ipython().system('pip install JSAnimation')

# custom utilies for displaying animation, collecting rollouts and more
import pong_utils

def run():
    # get_ipython().run_line_magic('matplotlib', 'inline')

    # check which device is being used. 
    # I recommend disabling gpu until you've made sure that the code runs
    device = pong_utils.device
    print("using device: ",device)


    # # In[2]:


    # render ai gym environment
    import gym
    import time

    # PongDeterministic does not contain random frameskip
    # so is faster to train than the vanilla Pong-v4 environment
    env = gym.make('PongDeterministic-v4')

    print("List of available actions: ", env.unwrapped.get_action_meanings())

    # we will only use the actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
    # the 'FIRE' part ensures that the game starts again after losing a life
    # the actions are hard-coded in pong_utils.py


    # # Preprocessing
    # To speed up training, we can simplify the input by cropping the images and use every other pixel
    # 
    # 

    # # In[3]:


    import matplotlib
    import matplotlib.pyplot as plt

    # show what a preprocessed image looks like
    env.reset()
    _, _, _, _ = env.step(0)
    # get a frame after 20 steps
    for _ in range(20):
        frame, _, _, _ = env.step(1)

    plt.subplot(1,2,1)
    plt.imshow(frame)
    plt.title('original image')

    plt.subplot(1,2,2)
    plt.title('preprocessed image')

    # 80 x 80 black and white image
    plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
    plt.show()


    # # Policy
    # 
    # ## Exercise 1: Implement your policy
    #  
    # Here, we define our policy. The input is the stack of two different frames (which captures the movement), and the output is a number $P_{\rm right}$, the probability of moving left. Note that $P_{\rm left}= 1-P_{\rm right}$

    # # In[4]:


    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np



    # set up a convolutional neural net
    # the output is the probability of moving right
    # P(left) = 1-P(right)
    class Policy(nn.Module):

        def __init__(self):
            super(Policy, self).__init__()
            
            
        ########
        ## 
        ## Modify your neural network
        ##
        ########
            
            # 80x80 to outputsize x outputsize
            # outputsize = (inputsize - kernel_size + stride)/stride 
            # (round up if not an integer)

            # # output = 20x20 here
            # self.conv = nn.Conv2d(2, 1, kernel_size=4, stride=4)
            # self.size=1*20*20
            
            # # 1 fully connected layer
            # self.fc = nn.Linear(self.size, 1)
            # self.sig = nn.Sigmoid()

            num_inputs = 80*80*2
            action_size = 1
            self.in_layer = nn.Conv2d(2, 64, 8, 4)
            # The second hidden layer convolves 64 filters of 4 x 4 with stride 2,
            #  followed by a rectifier nonlinearity
            self.conv2 = nn.Conv2d(64, 16, 4, 2)
            # third convolutional layer that convolves 64 filters of 3 x 3 with stride 1
            #  followed by a rectifier
            # self.conv3 = nn.Conv2d(32, 8, 3, 1)
            self.size=8*8*16
            # The final hidden layer is fully-connected and consists of 512 rectifier units.
            self.layer4 = nn.Linear(self.size, 256)
            self.out_layer = nn.Linear(256, action_size)
            # Sigmoid to 
            self.sig = nn.Sigmoid()

            
        def forward(self, x):
            
        ########
        ## 
        ## Modify your neural network
        ##
        ########
        
            x = F.relu(self.in_layer(x))
            x = F.relu(self.conv2(x))
            # x = F.relu(self.conv3(x))
            x = x.view(-1,self.size)
            x = self.layer4(x)
            x = self.out_layer(x)
            x = self.sig(x)
            return x

            # x = F.relu(self.conv(x))
            # # flatten the tensor
            # x = x.view(-1,self.size)
            # return self.sig(self.fc(x))

    # use your own policy!
    policy=Policy().to(device)
    # policy=pong_utils.Policy().to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    import torch.optim as optim
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)


    # # Game visualization
    # pong_utils contain a play function given the environment and a policy. An optional preprocess function can be supplied. Here we define a function that plays a game and shows learning progress

    # # In[6]:


    pong_utils.play(env, policy, time=100) 
    # try to add the option "preprocess=pong_utils.preprocess_single"
    # to see what the agent sees


    # # Rollout
    # Before we start the training, we need to collect samples. To make things efficient we use parallelized environments to collect multiple examples at once

    # # In[7]:


    envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
    prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=100)


    # # In[8]:


    print(reward)


    def clipped_surrogate(policy, old_probs, states, actions, rewards,
                        discount = 0.995, epsilon=0.1, beta=0.01):

        ########
        ## 
        ## WRITE YOUR OWN CODE HERE
        ##
        ########
        
        # create discounts
        discount = discount**np.arange(len(rewards))

        # discount rewards
        rewards = np.asarray(rewards)*discount[:,np.newaxis]

        # get only future rewards
        # ?? looks like all rewards are in the future???
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        # normalize rewards_future
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

        # prep as pytorch
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)
        
        # convert states to policy (or probability)
        new_probs = pong_utils.states_to_prob(policy, states)
        new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)

        ratio = new_probs/old_probs
        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_rewards = torch.min(ratio*rewards, clip*rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # prevents policy to become exactly 0 or 1 helps exploration
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
            (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_rewards + beta*entropy)

    Lsur= clipped_surrogate(policy, prob, state, action, reward)

    print(Lsur)


    # # Training
    # We are now ready to train our policy!
    # WARNING: make sure to turn on GPU, which also enables multicore processing. It may take up to 45 minutes even with GPU enabled, otherwise it will take much longer!

    # # In[ ]:


    from parallelEnv import parallelEnv
    import numpy as np
    # WARNING: running through all 800 episodes will take 30-45 minutes

    # training loop max iterations
    episode = 500
    # episode = 800

    # widget bar to display progress
    # get_ipython().system('pip install progressbar')
    import progressbar as pb
    widget = ['training loop: ', pb.Percentage(), ' ', 
            pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

    # # Initialize environment
    envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

    discount_rate = .99
    beta = .01
    tmax = 320

    # keep track of progress
    mean_rewards = []

    for e in range(episode):

        # collect trajectories
        old_probs, states, actions, rewards =         pong_utils.collect_trajectories(envs, policy, tmax=tmax)
            
        total_rewards = np.sum(rewards, axis=0)

        # this is the SOLUTION!
        # use your own surrogate function
        L = -clipped_surrogate(policy, old_probs, states, actions, rewards, beta=beta)
        # L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L
            
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995
        
        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))
        
        # display some progress every 20 iterations
        if (e+1)%20 ==0 :
            print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
            print(total_rewards)
            
        # update progress widget bar
        timer.update(e+1)
        
    timer.finish()
        


    # # In[15]:


    # play game after training!
    pong_utils.play(env, policy, time=2000) 


    # # In[ ]:


    plt.plot(mean_rewards)


    # # In[ ]:


    # save your policy!
    torch.save(policy.state_dict(), 'REINFORCE.policy')

    # load your policy if needed
    # policy = torch.load('REINFORCE.policy')

    # try and test out the solution!
    # policy = torch.load('PPO_solution.policy')

if __name__ == '__main__':
    run()