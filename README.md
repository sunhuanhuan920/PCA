# Principal Component Analysis

This is a python implementation of principal component analysis using both **eigen decomposition** and **singular value decomposition**

The comparison between sklearn's PCA implementation and my PCA implementation on iris dataset
![alt text](./Image/comparison.png)
As you can see from the image above that three principal component plots vary from each other, this is because the sign of principal components (eigenvectors) obtained by using sklearn's implementation of PCA and my implementation of PCA are different. 

However, since PCA is a mathematical transformation that aims to find axis that retain as much variance as possible from original data, and principal components are vectors that has different direction thus if I change the signs of the components, it does not change the variance that is contained in each component. Thus signs do not change the interpretation of PCA and therefore it doesn't matter if you end up with eigenvectors with different sign.