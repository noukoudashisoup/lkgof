"""Module containing Stein discrepancy-related classes"""
import numpy as np


def stein_kernel_gram(X, score, k):
    """
    Compute the Stein kernel gram matrix hp(x_i, x_j)
    Args: 
        - X: an n x d data numpy array
        - score: n x d numpy array; score evaluated at X
        - k: a KST/DKST object
    Return:
        - an n x n array
    """
    n, d = X.shape
    # print('Shape of X')
    # print(X.shape)
    # n x d matrix of gradients
    # n x n
    gram_score_p = score.dot(score.T)
    # print(np.sum(np.ones([n,n])[gram_score_p<0])/n**2-1./n)
    # n x n
    K = k.eval(X, X)

    B = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(d):
        score_p_i = score[:, i]
        B += k.gradX_Y(X, X, i) * score_p_i
        C += (k.gradY_X(X, X, i).T * score_p_i).T
    H = K*gram_score_p + B + C + k.gradXY_sum(X, X)
    return H


def minflow_stein_kernel_gram(X, W, k):
    """
    Compute the Stein kernel gram matrix hp(x_i, x_j)
    Args: 
        - X: an n x d data numpy array
        - W: n x d numpy array; the coefficient formed by the transtion rate
        - k: a KST/DKST object
    Return:
        - an n x n array
    """
    n, d = X.shape
    n_values = k.n_values
    H = np.zeros([n, n])
    K = k.eval(X, X)
    from lkgof.mcmc import score_shifts

    for i, shift in enumerate(score_shifts):
        for di in range(d):
            X1 = X.copy()
            outer = np.outer(W[i, :, di], W[i, :, di])
            X1[:, di] = np.mod(X[:, di]+shift, n_values[di])
            K1 = k.eval(X1, X)
            K2 = k.eval(X1, X1)
            kdiff = K2 + K - (K1 + K1.T)
            H += outer * kdiff

    return H

