""" 
    
    Principal Component Analysis
    
    Master in Computer Vision - Barcelona
    
    Week 1 - Task 4: Compress the descriptor, using PCA: add into mcv_bow.py and implement in mcv_pca.py
    
    Author: Francesco Ciompi, Ramon Baldrich, Jordi Gonzalez

         
"""

from PIL import Image
from numpy import *

#
# only projection is implemented here! so the step from projections to original data should be implemented
#
def pca(X):
    """ Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
    """
    # get dimensions
    num_data,dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order

        for i in range(V.shape[1]):
            V[:,i] /= S
        else:
            # PCA - SVD used
            U,S,V = linalg.svd(X)
            V = V[:num_data] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V,S,mean_X





