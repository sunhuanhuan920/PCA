"""
    Principal Component Analysis Class
        - Using regular eigenvalue and eigenvector decomposition
        - Using singular value decomposition (SVD)
    Arthor: Zhenhuan(Steven) Sun
"""

import numpy as np

class PCA():
    def __init__(self, n_components, method="eigen_decomposition"):
        # how many principal components user want to use
        self.n_components = n_components

        # the method we use to find the eigenvalues and eigenvectors
        self.method = method

    def fit(self, X):
        # number of examples and number of features
        self.n_examples, self.n_features = X.shape

        # compute the mean of each feature in the data
        self.mean = np.sum(X, axis=0) / self.n_examples

        # make sure that data has zero mean
        X = X - self.mean

        if self.method == "eigen_decomposition":
            # unbiasd covariance matrix of data
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

        elif self.method == "SVD":
            # perform singular value decomposition on data matrix X
            u, s, v = np.linalg.svd(X, full_matrices=True)

            # eigenvalues
            eigenvalues = (s ** 2) / (self.n_components - 1)

            # s is a vector thus we need to formalize singular value matrix 
            # using sigular values stored in s
            self.singular_value_matrix = np.identity(u.shape[1])
            for i in range(len(s)):
                self.singular_value_matrix[i] *= s[i]

            self.unitary_matrix = u
            self.explained_variance = eigenvalues[:self.n_components]
            self.explained_variance_ratio = (eigenvalues / eigenvalues.sum())[:self.n_components]
            self.components = v[:self.n_components, :]

        return self

    def transform(self, X):
        if self.method == "eigen_decomposition":
            # center the data
            X = X - self.mean
            # project original data onto principal components that maximize variance
            return X.dot(self.components.T)
        elif self.method == "SVD":
            # project original data onto principal componenets that maximize variance
            return self.unitary_matrix.dot(self.singular_value_matrix)