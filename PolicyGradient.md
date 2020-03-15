[expectedReturnFormal]: images/expectedReturnFormal.png "expected return"
[gradientAscentUpdateStep]:images/gradientAscentUpdateStep.png "gradient Ascent Update Step"
[ReinforcePseudocode01]:/images/ReinforcePseudocode01.png "ReinforcePseudocode01"
[ReinforcePseudocode02]:/images/ReinforcePseudocode02.png "ReinforcePseudocode02"
[ReinforcePseudocode03]:/images/ReinforcePseudocode03.png "ReinforcePseudocode03"
[ReinforcePseudocode04]:/images/ReinforcePseudocode04.png "ReinforcePseudocode04"
[ReinforceLikelihoodRatioPolicyGradient]:/images/ReinforceLikelihoodRatioPolicyGradient.png "ReinforceLikelihoodRatioPolicyGradient"
[BeyondReinforce01]:/images/BeyondReinforce01.png "BeyondReinforce01"
[BeyondReinforce02]:/images/BeyondReinforce02.png "BeyondReinforce02"
[BeyondReinforce03]:/images/BeyondReinforce03.png "BeyondReinforce03"
[NoiseReduction01]:/images/NoiseReduction01.png "NoiseReduction01"
[NoiseReduction02]:/images/NoiseReduction02.png "NoiseReduction02"
[RewardsNormalization01]:/images/RewardsNormalization01.png "RewardsNormalization01"
[CreditAssignment01]:/images/CreditAssignment01.png "CreditAssignment01"
[CreditAssignment02]:/images/CreditAssignment02.png "CreditAssignment02"
[CreditAssignment03]:/images/CreditAssignment03.png "CreditAssignment03"
tau

Theta


![expectedReturnFormal]

_(U,Theta, equals, sum over tau, probability of (tau, semi-colon, Theta.) multiplied by R,(tau))_

_(The expected return, U,Theta, equals the sum over all tau. With the probability of trajectory, tau, as influenced by, Theta, multiplied by the return from the arbitrary trajectory, tau)_

To see how it corresponds to the **expected return**, note that we've expressed the **return R(τ)** as a function of the trajectory τ. Then, we calculate the weighted average _(where the weights are given by_ P(τ;θ)) of all possible values that the return R(τ) can take.

_(To see how it corresponds to the expected return, note that we've expressed the return, R,(tau), as a function of the trajectory, tau. Then, we calculate the weighted average, (where the weights are given by the probability, P, of, (tau, semi-colon, theta)), of all possible values that the return, R,(tau), can take.)_

### REINFORCE

Our goal is to find the values of the weights θ in 
the neural network that maximize the expected return U

_(Our goal is to find the values of the weights, theta, in 
the neural network that maximize the expected return, U)_

One way to determine the value of θ that maximizes this function is through gradient ascent

_(One way to determine the value of theta the that maximizes this function is through gradient ascent)_

Our update step for gradient ascent appears as follows:

![gradientAscentUpdateStep]

_(Let theta equal, theta, plus, alpha, multiplied by, nabla theta, multiplied by, U,Theta )_

_(Let theta equal, theta, plus, alpha, multiplied by, column vector of partial derivatives, nabla, of theta, multiplied by, the expected return, U,Theta )_

where α is the step size that is generally allowed to decay over time. Once we know how to calculate or estimate this gradient, we can repeatedly apply this update step, in the hopes that θ converges to the value that maximizes U(θ).

_(where alpha is the step size that is generally allowed to decay over time. Once we know how to calculate or estimate this gradient, we can repeatedly apply this update step, in the hopes that theta converges to the value that maximizes U(theta).)_

#### Pseudocode

The algorithm described in the video is known as **REINFORCE**. The pseudocode is summarized below.

![ReinforcePseudocode01]

_(1. Use the policy, pi, theta, to collect, m, trajectories. {trajectory tau 1, trajectory tau 2, dot, dot, dot, trajectory tau, m,} with horizon, H. We refer to the eyth trajectory as)_

_(trajectory tau, (i), equals, ( state, s, at timestep, 0, i, action, a, at timestep, 0, i, dot, dot, dot, state, s, at timestep, horizon, H, i, and action, a, at timestep horizon, H, i, state, s, at timestep, horizon, H, +, 1, i))_

![ReinforcePseudocode02]

_(2. Use the trajectories to estimate the gradient column vector of partial derivatives, nabla, of theta, for the expected return, U,Theta )_

_(column vector of partial derivatives, nabla, of theta, for the expected return, U,Theta, is approximately equal to, the estimate of the gradient, g, hat. is defined by, 1 over m, sum over, m, starting at 1, sum over horizon, H, starting at 0. column vector of partial derivatives, nabla, of theta, multiplied by the log of policy, pi, theta, with the action a, at timestep t, i, and state, s, at timestep t, i) multiplied by the return from the arbitrary trajectory, tau))_

![ReinforcePseudocode03]

_(3. Update the weights of the policy)_

_(Let theta equal, theta, plus, alpha, multiplied by, the estimate of the gradient, g hat)_


![ReinforcePseudocode04]

_(4. Loop over steps 1 through 3)_


#### Likelihood Ratio Policy Gradient

We'll begin by exploring how to calculate the gradient ∇θ U(θ). The calculation proceeds as follows:

_(We'll begin by exploring how to calculate the gradient nabla, of theta, for the expected return, U. The calculation proceeds as follows:)_

![ReinforceLikelihoodRatioPolicyGradient]

_(First, we note line (1) follows directly from, 
The expected return, U,Theta, equals the sum over all tau. With the probability of trajectory, tau, as influenced by, Theta, multiplied by the return from the arbitrary trajectory, tau. where we've only taken the gradient of both sides)_

_(Then, we can get line (2) by just noticing that we can rewrite the gradient of the sum as the sum of the gradients.)_

_(In line (3), we only multiply every term in the sum by the probability of trajectory, tau, as influenced by theta, divided by, the probability of trajectory, tau, as influenced by theta. which is perfectly allowed because this fraction is equal to one!)_

_(Next, line (4) is just a simple rearrangement of the terms from the previous line. )_

_(Finally, line (5) follows from the chain rule, and the fact that the gradient of the log of a function is always equal to the gradient of the function, divided by the function)_

_(The final "trick" that yields line (5) (i.e., \nabla_\theta \log P(\tau;\theta) = \frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}∇ 
θ
​	 logP(τ;θ)= 
P(τ;θ)
∇ 
θ
​	 P(τ;θ)
​	 ) is referred to as the likelihood ratio trick or REINFORCE trick.)_

_(Likewise, it is common to refer to the gradient as the likelihood ratio policy gradient:)_

Once we’ve written the gradient as an expected value in this way, it becomes much easier to estimate.

### Beyond REINFORCE

Here, we briefly review key ingredients of the REINFORCE algorithm.

REINFORCE works as follows: First, we initialize a random policy πθ(a;s), and using the policy we collect a trajectory -- or a list of (state, actions, rewards) at each time step:

![BeyondReinforce01]

s1, a1, r1, s2, a2, r2,...

Second, we compute the total reward of the trajectory R=r1+r2+r3+..., and compute an estimate the gradient of the expected reward, 

![BeyondReinforce02]

_(the expected reward g, equals, the reward, r, sumed over all timesteps, t, column vector of partial derivatives, nabla, of theta, multiplied by the log of, policy, pi, theta, for the action, state, pair, a, s, at timestep, t)_

Third, we update our policy using gradient ascent with learning rate alpha:

![BeyondReinforce03]

_(Let theta equal, theta, plus, alpha, multiplied by, the estimate of the gradient, g)_

The process then repeats.

What are the main problems of REINFORCE? There are three issues:

1. The update process is very **inefficient!** We run the policy once, update once, and then throw away the trajectory.

2. The gradient estimate, _g_, is very **noisy**. By chance the collected trajectory may not be representative of the policy.

3. There is no clear **credit assignment**. A trajectory may contain many good/bad actions and whether these actions are reinforced depends only on the final total output.

In the following concepts, we will go over ways to improve the REINFORCE algorithm and resolve all 3 issues. All of the improvements will be utilized and implemented in the PPO algorithm.

### Noise Reduction

The way we optimize the policy is by maximizing the average rewards U(θ). To do that we use stochastic gradient ascent. Mathematically, the gradient is given by an average over all the possible trajectories,

_(The way we optimize the policy is by maximizing the average rewards, U,(theta). To do that we use stochastic gradient ascent. Mathematically, the gradient is given by an average over all the possible trajectories,)_

![NoiseReduction01]

_(average over all trajectories)_

_(only one is sampled)_

There could easily be well over millions of trajectories for simple problems, and infinite for continuous problems.

For practical purposes, we simply take one trajectory to compute the gradient, and update our policy. So a lot of times, the result of a sampled trajectory comes down to chance, and doesn't contain that much information about our policy. How does learning happen then? The hope is that after training for a long time, the tiny signal accumulates.

The easiest option to reduce the noise in the gradient is to simply sample more trajectories! Using distributed computing, we can collect multiple trajectories in parallel, so that it won’t take too much time. Then we can estimate the policy gradient by averaging across all the different trajectories

![NoiseReduction02]

### Rewards Normalization

There is another bonus for running multiple trajectories: we can collect all the total rewards and get a sense of how they are distributed.

In many cases, the distribution of rewards shifts as learning happens. Reward = 1 might be really good in the beginning, but really bad after 1000 training episode.

Learning can be improved if we normalize the rewards, where μ is the mean, and σ the standard deviation.

_(Learning can be improved if we normalize the rewards, where, mu, is the mean, and, sigma, the standard deviation.)_

![RewardsNormalization01]

_(let, the eyth reward, R, i, be, R, i, minus, mu, divided by, sigma)_

_(where, mu, equals, 1 over N, times, the sum over i, for the Reward of i)_

_(where, sigma, equals, the square root of, 1 over N, times, the sum over, i, (reward, R, i, minus, mu, squared))_

(when all the Ri are the same, σ=0, we can set all the normalized rewards to 0 to avoid numerical problems)

_(when all the, rewards, R,i, are the same, sigma, =, 0, we can set all the normalized rewards to 0 to avoid numerical problems)_

### Credit Assignment

Going back to the gradient estimate, we can take a closer look at the total reward, R, which is just a sum of reward at each step. R = r 1 + r 2 + ... + r t -1 + r t +... 

_(Going back to the gradient estimate, we can take a closer look at the total reward, R, which is just a sum of reward at each step. R =, r, at timestep 1, +, r, at timestep 2, +, dot dot dot, +, r at timestep t, -, 1, +, r at timestep, t, +, dot dot dot)_ 

![CreditAssignment01]

Let’s think about what happens at time-step tt. Even before an action is decided, the agent has already received all the rewards up until step t-1t−1. So we can think of that part of the total reward as the reward from the past. The rest is denoted as the future reward.

![CreditAssignment02]

_(R, past, equals, rewards up until, reward, r, at timestep, t, -, 1.)_

_(R, future, equals, rewards from, reward, r, at timestep, t, plus all future rewards)_

Because we have a Markov process, the action at time-step t can only affect the future reward, so the past reward shouldn’t be contributing to the policy gradient. So to properly assign credit to the action, a, at timestep, t, we should ignore the past reward. So a better policy gradient would simply have the future reward as the coefficient.

![CreditAssignment03]

_(the expected reward g, equals, sum over all timesteps, t. The reward, R, future, times, column vector of partial derivatives, nabla, of theta, multiplied by the log of, policy, pi, theta, for the action, state, pair, a, s, at timestep, t)_

### Notes on Gradient Modification

You might wonder, why is it okay to just change our gradient? Wouldn't that change our original goal of maximizing the expected reward?

It turns out that mathematically, ignoring past rewards might change the gradient for each specific trajectory, but it doesn't change the averaged gradient. So even though the gradient is different during training, on average we are still maximizing the average reward. In fact, the resultant gradient is less noisy, so training using future reward should speed things up!

