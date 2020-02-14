"""
Function for 'decoding' an input vector x_i (conditioned on a set of context points) to obtain a
prediction for y_i (mean and uncertainty). The conditioning is achieved by concatenating x_i with
the 'aggregated embedding', or 'context vector' r, which has been calculated by averaging the
encodings {r_c} of the context points. The function comprises a fully connected neural network,
with size and number of hidden layers being hyperparameters to be selected.

Input = (x_i, r). Output = (y_mean_i, y_var_i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

class Decoder(nn.Module):
    """The Decoder."""
    def __init__(self, input_size, output_size, decoder_n_hidden=2, decoder_hidden_size=8):
        """
        :param input_size: An integer describing the dimensionality of the input, in this case
                           (r_size + x_size), where x_size is the dimensionality of x and r_size
                           is the dimensionality of the embedding r
        :param output_size: An integer describing the dimensionality of the output, in this case
                            y_size, which is the dimensionality of the target variable y
        :param decoder_n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param decoder_hidden_size: An integer describing the number of nodes in each layer of the
                                    neural network
        """

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = decoder_n_hidden
        self.hidden_size = decoder_hidden_size

        self.fcs = nn.ModuleList()

        for i in range(decoder_n_hidden + 1):
            if i == 0:
                self.fcs.append(nn.Linear(input_size, decoder_hidden_size))

            elif i == decoder_n_hidden:
                self.fcs.append(nn.Linear(decoder_hidden_size, 2 * output_size))

            else:
                self.fcs.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))

    def forward(self, x, aggregated_embedding):
        """
        :param x: A tensor of dimensions [context_set_samples, N, x_size], containing the input
                  values x
        :param aggregated_embedding: A tensor of dimensions [context_set_samples, r_size]
                                     containing the context vectors r
        :return dist: The distributions over the predicted outputs y_target
        :return mu: A tensor of dimensionality [context_set_samples, N_target, output_size]
                    describing the means of the normal distribution
        :return var: A tensor of dimensionality [context_set_samples, N_target, output_size]
                     describing the variances of the normal distribution
        """

        context_set_samples = x.shape[0]

        #Concatenate the input vectors x_i to the aggregated embedding r.
        aggregated_embedding = torch.unsqueeze(aggregated_embedding, dim=1).repeat(1, x.shape[1],
                                                                                   1)
        x = torch.cat((x, aggregated_embedding), dim=2)
        #x = [context_set_samples, N, (x_size + r_size)]
        x = x.view(-1, self.input_size)
        #x = [context_set_samples * N, (x_size + r_size)]
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)
        #x = [context_set_samples * N, 2 * output_size]

        #The outputs are the predicted y means and variances
        mus, log_sigmas = x[:, :self.output_size], x[:, self.output_size:]
        sigmas = 0.00001 + 0.99999 * F.softplus(log_sigmas)
        #mu, sigma = [context_set_samples * N, output_size]

        mus = mus.view(context_set_samples, -1, self.output_size)
        #[context_set_samples, N, output_size]
        sigmas = sigmas.view(context_set_samples, -1, self.output_size)
        #[context_set_samples, N, output_size]

        dists = [MultivariateNormal(mu, torch.diag_embed(sigma)) for mu, sigma in
                 zip(mus, sigmas)]

        return dists, mus, sigmas
