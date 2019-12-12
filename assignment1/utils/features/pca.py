import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    # TODO: Implement PCA by extracting        #
    # eigenvector.You may need to sort the      #
    # eigenvalues to get the top K of them.     #
    ###############################################
    ###############################################
    #          START OF YOUR CODE         #
    ###############################################
    n,m = X.shape
    
    cov = np.dot(np.transpose(X),X)/(n-1)
    p_s, vectors = np.linalg.eig(cov)
    args = (np.argsort(p_s))[::-1]
    T = p_s[args[:K]]
    vectors = np.transpose(vectors)
    P = np.asarray([vectors[x] for x in args[:K]])
  
    
    
    

    ###############################################
    #           END OF YOUR CODE         #
    ###############################################
    
    return (P, T)
