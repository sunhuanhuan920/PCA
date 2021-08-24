from PCA import PCA as myPCA
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# normalization (0 mean and unit variance)
X_norm = StandardScaler().fit_transform(X)

# Using scikit-learn's PCA implementation
pca = PCA(n_components=4)
pca.fit(X_norm)
print("scikit-learn implementation of PCA: ")
print("explained variance ratio: \n", pca.explained_variance_ratio_)
print("components (eigenvectors): \n", pca.components_)

# project X onto directions which maximaize the variance (on principal components)
X_proj_1 = pca.transform(X_norm)

# using my implementation of PCA with eigen decomposition
my_pca = myPCA(n_components=4, method="eigen_decomposition")
my_pca.fit(X_norm)
print("my implementation of PCA (eigen decomposition): ")
print("explained variance ratio: \n", my_pca.explained_variance_ratio)
print("components (eigenvectors): \n", my_pca.components)

# project X onto directions which maximaize the variance (on principal components)
X_proj_2 = my_pca.transform(X_norm)

# using my implementation of PCA with svd
my_pca_svd = myPCA(n_components=4, method="SVD")
my_pca_svd.fit(X_norm)
print("my implementation of PCA (SVD): ")
print("explained variance ratio: \n", my_pca_svd.explained_variance_ratio)
print("components (eigenvectors): \n", my_pca_svd.components)

# project X onto directions which maximaize the variance (on principal components)
X_proj_3 = my_pca_svd.transform(X_norm)

# plot the first two pricipal components for visualization
fig, axes = plt.subplots(1, 3)
axes[0].scatter(X_proj_1[:, 0], X_proj_1[:, 1], c=y)
axes[0].set_title("scikit-learn PCA")
axes[1].scatter(X_proj_2[:, 0], X_proj_2[:, 1], c=y)
axes[1].set_title("my PCA using eigen decomposition")
axes[2].scatter(X_proj_3[:, 0], X_proj_3[:, 1], c=y)
axes[2].set_title("my PCA using SVD")

for ax in axes.flat:
    ax.set(xlabel="Principal Component 1", ylabel="Principal Component 2")

plt.show()