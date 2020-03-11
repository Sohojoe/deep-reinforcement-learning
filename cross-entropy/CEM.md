

# Udacity Reinforcement Learning Nanodagree - Project One
Joe Booth Febuary 2019

## Document Objective
The goal of this document is to provide a fairly accessible overview of the Cross Entropy Method (CEM) for Reinforcement Learning.

Note: This report aims to be accessible via text to speech browser plugins. To that end, I have phonetically typed out the 'alt-text' tag under equations (typically in brackets). I also use spaces between letters so that the text to speech plugin correctly pronounces the phrase. For example, I denote state at timestep t, as, 's t', as opposed to, 'st'

This report covers the following areas:

* An overview of the Cross Entropy Method (CEM) algorithm. 
* The implementation of the DQN Algorithm, including model architecture and hyperparameters.
* Conclusion

## The Cross Entropy Method
Cross Entropy Method

`pop_size` = number of samples to explore each step
`sigma` = standard diviation of noise to apply to the best_weight

foreach step:
    foreach pop_size:
        current_weights <- best_weight + random*`sigma`
        evaluate over one episode using current_weights
        append reward to rewards
    
    take the weights from the 10 best rewards
    get the mean of these 10 weights
    this becomes new best_weight
    evaluate new best_weight for one episode
    



