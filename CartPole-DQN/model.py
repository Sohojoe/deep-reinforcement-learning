import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        num_inputs = state_size
        self.in_layer = nn.Linear(num_inputs, 32)
        self.hidden_1 = nn.Linear(32, 32)
        self.out_layer = nn.Linear(32, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.in_layer(state))
        x = F.relu(self.hidden_1(x))
        x = self.out_layer(x)
        return x


class CnnQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        # The input to the neural network consists of an 84 x 84 x 4 image 

        # The first hidden layer convolves 32 filters of 8 x 8 with stride 4
        #  with the input image and applies a rectifier nonlinearity
        # num_inputs = 84*84*4
        num_inputs = state_size
        self.in_layer = nn.Conv2d(num_inputs, 64, 8, 4)

        # The second hidden layer convolves 64 filters of 4 x 4 with stride 2,
        #  followed by a rectifier nonlinearity
        self.conv2 = nn.Conv2d(64, 32, 4, 2)

        # third convolutional layer that convolves 64 filters of 3 x 3 with stride 1
        #  followed by a rectifier
        self.conv3 = nn.Conv2d(32, 20, 3, 1)

        # The final hidden layer is fully-connected and consists of 512 rectifier units.
        self.layer4 = nn.Linear(20*11*11, 512)
        self.out_layer = nn.Linear(512, action_size)

        # The output layer is a fully-connected linear layer with a
        #  single output for each valid action.
        #  The number of valid actions varied between 4 and 18 on the games we considered.

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.in_layer(state))
        x = F.relu(self.conv2(state))
        x = F.relu(self.conv3(state))
        x = self.layer4(state)
        x = self.out_layer(state)
        return x
