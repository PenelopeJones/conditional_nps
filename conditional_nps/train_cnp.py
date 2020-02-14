"""
Script for training a Conditional Neural Process on fragment representations of molecules.

Based on the work carried out in this paper:
Conditional Neural Processes: Garnelo M, Rosenbaum D, Maddison CJ, Ramalho T, Saxton D, Shanahan M,
Teh YW, Rezende DJ, Eslami SM. Conditional Neural Processes. In International Conference on Machine
Learning 2018.
"""
import time
import warnings
import argparse
import sys

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from active_learning.data_utils import parse_dataset, transform_data
from cnp.cnp_model import CNP

sys.path.append('../')
sys.path.append('../CNP/')


FREESOLV_PATH = '~/ml_physics/GraphChem/data/processed/FreeSolv_SAMPL.csv'
ESOL_PATH = '~/ml_physics/GraphChem/data/orig/ESOL_delaney-processed.csv'
CATS_PATH = '~/ml_physics/GraphChem/data/orig/CatS.csv'
MELTING_PATH = '~/ml_physics/GraphChem/data/orig/Meltingpoint.csv'
MALARIA_PATH = '~/ml_physics/GraphChem/data/orig/Malaria.csv'
QM9_PATH = '~/ml_physics/GraphChem/data/orig/qm9.csv'
CEP_PATH = '~/ml_physics/GraphChem/data/orig/CEP_pce.csv'

def main(task, feat, n_com, context_set_samples, learning_rate, iterations, r_size,
         encoder_hidden_size, encoder_n_hidden, decoder_hidden_size, decoder_n_hidden):
    """
    :param task: String, comprising the name of the dataset being investigated
    :param feat: String, describing how each molecule should be represented. If 'fragments',
                 then RDKit Fragments will be used to represent the molecules; else, Morgan
                 fingerprints will be used
    :param n_com: Integer, describing the number of PCA components that should be included
                  in the representation of the molecule
    :param context_set_samples: Integer, describing the number of times we should sample the set
                                of context points used to form the aggregated embedding during
                                training, given the number of context points to be sampled
                                N_context. When testing this is set to 1
    :param learning_rate: A float number, describing the optimiser's learning rate
    :param iterations: An integer, describing the number of iterations. In this case it also
                       corresponds to the number of times we sample the number of context points
                       N_context

    :param r_size: An integer describing the dimensionality of the embedding / context vector r
    :param encoder_hidden_size: An integer describing the number of nodes per hidden layer in the
                                encoder neural network
    :param encoder_n_hidden: An integer describing the number of hidden layers in the encoder neural
                             network
    :param decoder_hidden_size: An integer describing the number of nodes per hidden layer in the
                                decoder neural network
    :param decoder_n_hidden: An integer describing the number of hidden layers in the decoder neural
                             network
    :return:
    """

    warnings.filterwarnings('ignore')

    #Indicate whether fragments should be used to represent the molecules
    use_frag = True
    if feat != 'fragments':
        use_frag = False

    print('\nTraining CNP on ' + task + ' dataset with ' + feat + ' features (' + str(n_com) +
          ' components)')

    print('\nGenerating features...')
    if task == 'FreeSolv':
        X, y = parse_dataset(task, FREESOLV_PATH, use_frag)
    elif task == 'ESOL':
        X, y = parse_dataset(task, ESOL_PATH, use_frag)
    elif task == 'CEP':
        X, y = parse_dataset(task, CEP_PATH, use_frag)
    elif task == 'CatS':
        X, y = parse_dataset(task, CATS_PATH, use_frag)
    elif task == 'Melt':
        X, y = parse_dataset(task, MELTING_PATH, use_frag)
    elif task == 'Malaria':
        X, y = parse_dataset(task, MALARIA_PATH, use_frag)
    else:
        raise Exception('Must provide dataset')

    #If fragments are not used to represent the molecules, we will use Morgan fingerprints instead.
    if feat == 'fingerprints':
        rdkit_mols = [MolFromSmiles(smiles) for smiles in X]
        X = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in rdkit_mols]
        X = np.asarray(X)

    r2_list = []
    rmse_list = []
    mae_list = []
    time_list = []
    print('\nBeginning training loop...')
    j = 0
    for i in range(5, 10):
        start_time = time.time()

        #Randomly split the data into train and test sets, then standardise to zero mean and unit
        # variance.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        X_train, y_train, X_test, y_test, y_scaler = transform_data(X_train, y_train, X_test,
                                                                    y_test, n_com)

        #Convert the data for use in PyTorch.
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

        print('... building model.')

        #Build the Conditional Neural Process model, with the following architecture:
        #(x, y)_i --> encoder --> r_i
        #r = average(r_i)
        #(x*, r) --> decoder --> y_mean*, y_var*
        #The encoder and decoder functions are neural networks, with size and number of layers being
        # hyperparameters to be selected.
        cnp = CNP(x_size=X_train.shape[1], y_size=y_train.shape[1], r_size=r_size,
                  encoder_hidden_size=encoder_hidden_size, encoder_n_hidden=encoder_n_hidden,
                  decoder_hidden_size=decoder_hidden_size, decoder_n_hidden=decoder_n_hidden)

        print('... training.')

        #Train the model(NB can replace x_test, y_test with x_valid and y_valid if planning to use
        # a cross validation set)
        cnp.train(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test,
                  context_set_samples=context_set_samples, lr=learning_rate,
                  iterations=iterations, testing=True)

        #Testing: the 'context points' when testing are the entire training set, and the 'target
        # points' are the entire test set.
        x_context = torch.unsqueeze(X_train, dim=0)
        y_context = torch.unsqueeze(y_train, dim=0)
        x_test = torch.unsqueeze(X_test, dim=0)

        #Predict mean and error in y given the test inputs x_test
        _, predict_test_mean, predict_test_var = cnp.predict(x_context, y_context, x_test)
        x_test = torch.squeeze(x_test, dim=0)
        predict_test_mean = np.squeeze(predict_test_mean.data.numpy(), axis=0)
        predict_test_var = np.squeeze(predict_test_var.data.numpy(), axis=0)

        # We transform the standardised predicted and actual y values back to the original data
        # space
        y_mean_pred = y_scaler.inverse_transform(predict_test_mean)
        y_var_pred = y_scaler.var_ * predict_test_var
        y_test = y_scaler.inverse_transform(y_test)

        #Calculate relevant metrics
        score = r2_score(y_test, y_mean_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_mean_pred))
        mae = mean_absolute_error(y_test, y_mean_pred)

        time_taken = time.time() - start_time

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))
        print("Execution time: {:.3f}".format(time_taken))
        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)
        time_list.append(time_taken)

        np.savetxt('results/cnp/' + task + '_seed_' + str(j) + '_ypred_' + feat + '.dat',
                   y_mean_pred)
        np.savetxt('results/cnp/' + task + '_seed_' + str(j) + '_ytest.dat', y_test)
        np.savetxt('results/cnp/' + task + '_seed_' + str(j) + '_ystd_' + feat + '.dat',
                   np.sqrt(y_var_pred))
        j += 1

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    time_list = np.array(time_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list),
                                                np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list),
                                               np.std(rmse_list) / np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list),
                                                np.std(mae_list) / np.sqrt(len(mae_list))))
    print("mean Execution time: {:.3f} +- {:.3f}\n".format(np.mean(time_list),
                                                           np.std(time_list)/
                                                           np.sqrt(len(time_list))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ESOL',
                        help="Dataset on which to train the GP. Possible task names: Melt, ESOL, "
                             "FreeSolv_SAMPL, QM9, CEP, CatS, Malaria")
    parser.add_argument('-feat', type=str, default='fragments',
                        help='Molecular representation used as features.')
    parser.add_argument('-n_com', type=int, default=64,
                        help='Number of PCA components used as the input features.')
    parser.add_argument('--context_set_samples', type=int, default=64,
                        help='The number of samples to take of the context set, given the number of'
                             ' context points that should be selected.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The training learning rate.')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Number of training iterations.')
    parser.add_argument('--r_size', type=int, default=128,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('--encoder_hidden_size', type=int, default=8,
                        help='Dimensionality of encoder hidden layers.')
    parser.add_argument('--encoder_n_hidden', type=int, default=4,
                        help='Number of encoder hidden layers.')
    parser.add_argument('--decoder_hidden_size', type=int, default=8,
                        help='Dimensionality of decoder hidden layers.')
    parser.add_argument('--decoder_n_hidden', type=int, default=2,
                        help='Number of decoder hidden layers.')
    args = parser.parse_args()

    main(args.task, args.feat, args.n_com, args.context_set_samples, args.learning_rate,
         args.iterations, args.r_size, args.encoder_hidden_size, args.encoder_n_hidden,
         args.decoder_hidden_size, args.decoder_n_hidden)
