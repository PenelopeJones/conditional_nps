"""
Function for 'encoding' context points (x, y)_i using a fully connected neural network.
Input = (x, y)_i; output = r_i.
"""

import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """The Encoder."""
    def __init__(self, input_size, output_size, encoder_n_hidden=4, encoder_hidden_size=8):
        """
        :param input_size: An integer describing the dimensionality of the input to the encoder;
                           in this case the sum of x_size and y_size
        :param output_size: An integer describing the dimensionality of the embedding, r_i
        :param encoder_n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param encoder_hidden_size: An integer describing the number of nodes in each layer of
                                    the neural network
        """

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = encoder_n_hidden
        self.hidden_size = encoder_hidden_size
        self.fcs = nn.ModuleList()

        for i in range(encoder_n_hidden + 1):
            if i == 0:
                self.fcs.append(nn.Linear(input_size, encoder_hidden_size))

            elif i == encoder_n_hidden:
                self.fcs.append(nn.Linear(encoder_hidden_size, output_size))

            else:
                self.fcs.append(nn.Linear(encoder_hidden_size, encoder_hidden_size))

    def forward(self, x):
        """
        :param x: A tensor of dimensions [context_set_samples, number of context points
                  N_context, x_size + y_size]. In this case each value of x is the concatenation
                  of the input x with the output y (confusingly)
        :return: The embeddings, a tensor of dimensionality [context_set_samples, N_context,
                 r_size]
        """

        #Pass (x, y)_i through the fully connected neural network.
        x = x.view(-1, self.input_size)
        for fc in self.fcs:
            x = F.relu(fc(x))
        return x
