"""
Utility functions for active learning experiments
"""
import os

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, Descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_dataset(task_name, path, use_fragments=True, subset_size=1000):
    """
    Returns list of molecular smiles, as well as the y-targets of the dataset
    :param task_name: name of the task
    :param path: dataset path
    :param use_fragments: If True return fragments instead of SMILES
    :param subset_size: Subset size for big datasets like CEP or QM9
    :return: x, y where x can be SMILES or fragments and y is the label.
    """

    smiles_list = []
    y = None

    if task_name == 'FreeSolv':
        df = pd.read_csv(path)
        smiles_list = df['smiles'].tolist()
        y = df['expt'].to_numpy()  # can change to df['calc'] for calculated values

    if task_name == 'ESOL':
        df = pd.read_csv(path)
        smiles_list = df['smiles'].tolist()
        y = df['measured log solubility in mols per litre'].to_numpy()

    if task_name == 'CatS':
        df = pd.read_csv(path, usecols=[0, 1], header=None, names=['smiles', 'y'])
        smiles_list = df['smiles'].tolist()
        y = df['y'].to_numpy()

    if task_name == 'Melt':

        df = pd.read_csv(path)
        smiles_list = df['smiles'].tolist()
        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        good_inds = []
        good_mols = []

        # There are 3025/3042 molecules that can be parsed by RDKit. 3025 is the dataset size commonly reported in the
        # literature cf. the paper:
        # "Bayesian semi-supervised learning for uncertainty-calibrated prediction of molecular properties and
        # active learning"

        for ind, mol in enumerate(rdkit_mols):
            if mol != None:
                good_inds.append(ind)
                good_mols.append(mol)
        df = df.iloc[good_inds]
        smiles_list = df['smiles'].tolist()
        y = df['mpC'].to_numpy()

    if task_name == 'Malaria':
        df = pd.read_csv(path, header=0)
        pred_val = 'XC50_3D7 (microM)'
        df = df[((df[pred_val] != 'ND') & (df[pred_val] != '<')) & (df[pred_val].notnull())]
        smiles_list = df['smiles'].tolist()
        y = df[pred_val].to_numpy()
        y = np.log10(y)

    if task_name == 'CEP':
        df = pd.read_csv(path)
        df = df.sample(subset_size)
        smiles_list = df[df.columns[0]].tolist()

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        good_inds = []
        good_mols = []
        for ind, mol in enumerate(rdkit_mols):
            if mol != None:
                good_inds.append(ind)
                good_mols.append(mol)
        df = df.iloc[good_inds]

        smiles_list = df[df.columns[0]].tolist()
        y = df[df.columns[1]].to_numpy()

        if task_name == 'QM9':
            df = pd.read_csv(path)
            df = df.sample(
                subset_size)  # sets the dataframe as a random sample of the dataset with size subset_size (passed into parse_dataset as an arg)
            smiles_list = df['smiles'].tolist()
            # * A, GHz, Rotational Constant
            # * B, GHz, Rotational Constant
            # * C, GHz, Rotational Constant
            # * mu, D, Dipole moment
            # * alpha, a^{3}_{0}, Isotropic polarisability
            # * homo, Ha, Energy of HOMO
            # * lumo, Ha, Energy of LUMO
            # * gap, Ha, Gap (Energy of LUMO - Energy of HOMO)
            # * r2, a^{2}_0, Electronic Spatial Extent
            # * zpve, Ha, Zero point Vibrational energy
            # * u0, Ha, Internal Energy at 0K
            # * u298, Ha, Internal Energy at 298.15K
            # * h298, Ha, Enthalpy at 298.15K
            # * g298, Ha, Free Energy at 298.15K
            # * cv, cal/molK, Heat Capacity at 298.15K

            rdkit_mols = [MolFromSmiles(smiles) for smiles in
                          smiles_list]  # just using MolFromSmiles as a way of checking if rdkit can parse the smiles, and if it can, then it is a good string and keep it in the dataframe.
            good_inds = []
            good_mols = []
            for ind, mol in enumerate(rdkit_mols):  # checking for empty's or NaN's and skipping over those.
                if mol != None:
                    good_inds.append(ind)
                    good_mols.append(mol)
            df = df.iloc[
                good_inds]  # selecting data by integer location, df.iloc[<row_selection>, <column_selection>], removes bad_mols rows (i.e. empty or NaNs).

            # reset smiles_list to new subset smiles_list as it has changed
            smiles_list = df['smiles'].tolist()
            y = df['cv'].to_numpy()  # just using heat capacity as a test

    if use_fragments:

        # descList[115:] contains fragment-based features only (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)

        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        x = np.zeros((len(smiles_list), len(fragments)))
        for i in range(len(smiles_list)):
            mol = MolFromSmiles(smiles_list[i])
            try:
                features = [fragments[d](mol) for d in fragments]
            except:
                raise Exception('molecule {}'.format(i) + ' is not canonicalised')
            x[i, :] = features
    else:
        x = smiles_list

    return x, y


def transform_data(X_train, y_train, X_test, y_test, n_components):
    """
    Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :param y_train: train labels
    :param X_test: input test data
    :param y_test: test labels
    :param n_components: number of principal components to keep
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    pca = PCA(n_components)
    X_train_scaled = pca.fit_transform(X_train_scaled)
    print('Fraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
    X_test_scaled = pca.transform(X_test_scaled)
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
