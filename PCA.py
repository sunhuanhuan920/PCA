"""
    Principal Component Analysis Class
        - Using regular eigenvalue and eigenvector decomposition
        - Using singular value decomposition (SVD)
    Arthor: Zhenhuan(Steven) Sun
"""

import numpy as np

class PCA():
    def __init__(self, n_components):
        # how many principal components user want to use
        self.n_components = n_components

    def fit(self, X):
        # number of examples and number of features
        self.n_examples, self.n_features = X.shape

        # compute the mean of each feature in the data
        self.mean = np.sum(X, axis=0) / self.n_examples

        # make sure that data has zero mean
        X = X - self.mean

        # biasd covariance matrix of data
        covariance_matrix = X.T.dot(X) / (self.n_examples - 1)

        # compute the eigenvalue and eigenvector using covariance matrix
        # I used np.linalg.eigh() here instead of np.linalg.eig to avoid having
        # complex number as eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # check the correctness of eigenvalues and eigenvectors using A * u = lamda * u
        for i in range(len(eigenvalues)):
            eigenvector = eigenvectors[:, i].reshape(covariance_matrix.shape[1], 1)
            np.testing.assert_array_almost_equal(covariance_matrix.dot(eigenvector), eigenvalues[i] * eigenvector,
                                                decimal=3, err_msg="eigenvalue and eigenvector don't match", verbose=True)
            
        # sort the eigenvalues and eigenvectors pair in decreasing order
        eigen_pair_list= [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eigen_pair_list.sort(key=lambda x: x[0], reverse=True)

        self.explained_variance = np.array([x[0] for x in eigen_pair_list])[:self.n_components]
        self.explained_variance_ratio = (self.explained_variance / self.explained_variance.sum())[:self.n_components]
        self.components = np.array([x[1] for x in eigen_pair_list])[:self.n_components, :]

        return self

    def transform(self, X):
        # center the data
        X = X - self.mean

        return X.dot(self.components.T)
