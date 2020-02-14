"""
Conditional Neural Process (CNP): CNPs bridge the gap between neural networks and Gaussian
processes, allowing a distribution over functions to be learned and enabling the uncertainty
in a prediction to be estimated. They scale as O(n + m) where n is the number of training
points and m is the number of test points, in contrast to exact GPs which scale as O((n+m)^3).

Based on the work carried out in this paper:
Conditional Neural Processes: Garnelo M, Rosenbaum D, Maddison CJ, Ramalho T, Saxton D,
Shanahan M, Teh YW, Rezende DJ, Eslami SM. Conditional Neural Processes. In International
Conference on Machine Learning 2018.

"""

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from encoder import Encoder
from decoder import Decoder

class CNP():
    """
    The Conditional Neural Process model.
    """
    def __init__(self, x_size, y_size, r_size, encoder_hidden_size,
                 encoder_n_hidden, decoder_hidden_size, decoder_n_hidden):
        """
        :param x_size: An integer describing the dimensionality of the input x
        :param y_size: An integer describing the dimensionality of the target variable y
        :param r_size: An integer describing the dimensionality of the embedding / context
                       vector r
        :param encoder_hidden_size: An integer describing the number of nodes per hidden
                                    layer in the encoder neural network
        :param encoder_n_hidden: An integer describing the number of hidden layers in the
                                 encoder neural network
        :param decoder_hidden_size: An integer describing the number of nodes per hidden
                                    layer in the decoder neural network
        :param decoder_n_hidden: An integer describing the number of hidden layers in the
                                 decoder neural network
        """

        self.x_size = x_size
        self.y_size = y_size
        self.r_size = r_size
        self.encoder = Encoder((x_size + y_size), r_size, encoder_n_hidden,
                               encoder_hidden_size)
        self.decoder = Decoder((x_size + r_size), y_size, decoder_n_hidden,
                               decoder_hidden_size)
        self.optimiser = optim.Adam(list(self.encoder.parameters()) +
                                    list(self.decoder.parameters()))

    def train(self, x_train, y_train, x_test, y_test, x_scaler, y_scaler, context_set_samples, lr,
              iterations, testing, plotting):
        """
        :param x_train: A tensor with dimensions [N_train, x_size] containing the training
                        data (x values)
        :param y_train: A tensor with dimensions [N_train, y_size] containing the training
                        data (y values)
        :param x_test: A tensor with dimensions [N_test, x_size] containing the test data
                       (x values)
        :param y_test: A tensor with dimensions [N_test, y_size] containing the test data
                       (y values)
        :param x_scaler: The standard scaler used when testing == True to convert the
                         x values back to the correct scale.
        :param y_scaler: The standard scaler used when testing == True to convert the predicted
                         y values back to the correct scale.
        :param context_set_samples: An integer describing the number of times we should
                                    sample the set of context points used to form the
                                    aggregated embedding during training, given the number
                                    of context points to be sampled N_context. When testing
                                    this is set to 1
        :param lr: A float number, describing the optimiser's learning rate
        :param iterations: An integer, describing the number of iterations. In this case it
                           also corresponds to the number of times we sample the number of
                           context points N_context
        :param testing: A Boolean object; if set to be True, then every 30 iterations the
                        R^2 score and RMSE values will be calculated and printed for
                        both the train and test data
        :return:
        """

        self._context_set_samples = context_set_samples
        self._max_num_context = x_train.shape[0]
        self.iterations = iterations

        # At prediction time the context points comprise the entire training set.
        x_tot_context = torch.unsqueeze(x_train, dim=0)
        y_tot_context = torch.unsqueeze(y_train, dim=0)

        for iteration in range(iterations):
            self.optimiser.zero_grad()

            # Randomly select the number of context points N_context (uniformly from 3 to
            # N_train)
            num_context = torch.randint(low=1, high=self._max_num_context, size=(1,))
            # Randomly select N_context context points from the training data, a total of
            # context_set_samples times.
            idx = [np.random.permutation(self._max_num_context)[:num_context] for i in
                   range(self._context_set_samples)]
            x_context = [x_train[idx[i], :] for i in range(self._context_set_samples)]
            x_context = torch.stack(x_context)
            y_context = [y_train[idx[i], :] for i in range(self._context_set_samples)]
            y_context = torch.stack(y_context)

            # During training, we predict the y values for the entire training set and use
            # this to calculate the loss
            x_target = x_train.repeat(self._context_set_samples, 1, 1)
            y_target = y_train.repeat(self._context_set_samples, 1, 1)

            # The input to the encoder is (x, y)_i for all data points in the set of context
            # points. The encoder outputs an embedding r_i.
            input_context = torch.cat((x_context, y_context), dim=2)
            embed_context = self.encoder.forward(
                input_context.float())
            # embed_context = [context_set_samples*N_context, r_size]

            # The 'context' vector r is the average of these embeddings
            aggregated_embedding = self._aggregate(embed_context.float(),
                                                   self._context_set_samples)
            # aggregated_embedding = [context_set_samples, r_size]

            # The input to the decoder is the concatenation of the target x values and the
            # aggregated embedding;
            # the output is the distribution over y for each value of x (here we output mean,
            # variance).
            dists, _, _ = self.decoder.forward(x_target.float(), aggregated_embedding.float())

            # Calculate the loss
            log_ps = [dist.log_prob(y_target[i, ...].float()) for i, dist in enumerate(dists)]
            log_ps = torch.cat(log_ps)
            loss = -torch.mean(log_ps)
            self.losslogger = loss

            # The loss should generally decrease with number of iterations, though it is not
            # guaranteed to decrease monotonically because at each iteration the set of
            # context points changes randomly.
            if iteration % 100 == 0:
                print("Iteration " + str(iteration) + ":, Loss = " + str(loss.item()))

                # We can set testing = True if we want to check that we are not overfitting.
                if testing:
                    _, predict_train_mean, predict_train_var = self.predict(x_tot_context,
                                                                            y_tot_context,
                                                                            x_tot_context)
                    predict_train_mean = np.squeeze(predict_train_mean.data.numpy(), axis=0)
                    predict_train_var = np.squeeze(predict_train_var.data.numpy(), axis=0)

                    x_test = torch.unsqueeze(x_test, dim=0)
                    _, predict_test_mean, predict_test_var = self.predict(x_tot_context,
                                                                          y_tot_context,
                                                                          x_test)
                    x_test = torch.squeeze(x_test, dim=0)
                    predict_test_mean = np.squeeze(predict_test_mean.data.numpy(), axis=0)
                    predict_test_var = np.squeeze(predict_test_var.data.numpy(), axis=0)

                    # We transform the standardised predicted and actual y values back to the original data
                    # space
                    y_train_mean_pred = y_scaler.inverse_transform(predict_train_mean)
                    y_train_var_pred = y_scaler.var_ * predict_train_var
                    y_train_untransformed = y_scaler.inverse_transform(y_train)

                    # We transform the standardised predicted and actual y values back to the original data
                    # space
                    y_test_mean_pred = y_scaler.inverse_transform(predict_test_mean)
                    y_test_var_pred = y_scaler.var_ * predict_test_var
                    y_test_untransformed = y_scaler.inverse_transform(y_test)

                    if iteration % 1000 ==0:
                        if plotting:
                            x_c = x_scaler.inverse_transform(np.array(x_train))
                            y_c = y_train_untransformed
                            x_t = x_scaler.inverse_transform(np.array(x_test))
                            y_t = y_test_untransformed

                            plt.figure(figsize=(6, 6))
                            plt.scatter(x_c, y_c, color='black', s=10, marker='+', label="Context points")
                            plt.plot(x_t, y_t, linewidth=1, color='red', label="Target function")
                            plt.plot(x_t, y_test_mean_pred, color='darkcyan', linewidth=1, label='Predicted mean')
                            plt.fill_between(x_t[:, 0], y_test_mean_pred[:, 0] - 1.96 * np.sqrt(y_test_var_pred[:, 0]),
                                             y_test_mean_pred[:, 0] + 1.96 * np.sqrt(y_test_var_pred[:, 0]),
                                             color='cyan', alpha=0.2)
                            plt.title('r_size = ' + str(self.r_size) + ', ehs = ' + str(self.encoder.hidden_size)
                                      + ', enh = ' + str(self.encoder.n_hidden)
                                      + ', dhs = ' + str(self.decoder.hidden_size)
                                      + ', dnh = ' + str(self.decoder.n_hidden)
                                      + ', css = ' + str(self._context_set_samples))
                            plt.legend()
                            plt.savefig('cnp_1dreg' + str(iteration) + '.png')

                    r2_train = r2_score(y_train_untransformed, y_train_mean_pred)
                    rmse_train = mean_squared_error(y_train_untransformed, y_train_mean_pred)

                    r2_test = r2_score(y_test_untransformed, y_test_mean_pred)
                    rmse_test = mean_squared_error(y_test_untransformed, y_test_mean_pred)

                    print("R2 score (train) = " + str(r2_train))
                    print("R2 score (test) = " + str(r2_test))
                    print("RMSE (train) = " + str(rmse_train))
                    print("RMSE (test) = " + str(rmse_test))

            loss.backward()
            self.optimiser.step()

    def predict(self, x_context, y_context, x_target):
        """
        :param x_context: A tensor of dimensions [context_set_samples, N_context, x_size].
                          When training N_context is randomly sampled between 3 and N_train;
                          when testing N_context = N_train
        :param y_context: A tensor of dimensions [context_set_samples, N_context, y_size]
        :param x_target: A tensor of dimensions [N_target, x_size]
        :return dist: The distributions over the predicted outputs y_target
        :return mu: A tensor of dimensionality [context_set_samples, N_target, output_size]
                    describing the means
                    of the normal distribution.
        :return var: A tensor of dimensionality [context_set_samples, N_target, output_size]
                     describing the variances of the normal distribution.
        """

        train_embeddings = self.encoder.forward(torch.cat((x_context, y_context),
                                                          dim=2).float())
        context_set_samples = x_context.shape[0]
        aggregated_embedding = self._aggregate(train_embeddings.float(), context_set_samples)
        dist, mu, sigma = self.decoder.forward(x_target.float(), aggregated_embedding.float())
        return dist, mu, sigma

    def _aggregate(self, embedding, context_set_samples):
        """
        :param embedding: A tensor of dimensions [context_set_samples x N_context, r_size]
                          containing the embeddings of all context points
        :param context_set_samples: An integer describing the number of times that we have
                                    sampled the set of context points used to form the
                                    aggregated embedding during training, given the number
                                    of context points to be sampled N_context.


        :return: For each set of sampled context points, the average embedding is returned,
        which is a tensor of dimensions [context_set_samples, r_size]
        """

        embedding = embedding.view(context_set_samples, -1, self.r_size)
        return torch.squeeze(torch.mean(embedding, dim=1), dim=1)
