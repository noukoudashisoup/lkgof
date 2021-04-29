# all utility functions in kgof.util are visible.
from kgof.util import *
import numpy as np


def onehot_encode(X, n_values):
    """
    The input data X is assumed to be integer (string is now allowed)
    """
    n, d = X.shape
    U = np.unique(X)
    if len(U) > n_values:
        raise ValueError(('X contains more'
                         'than {} values').format(n_values))
    X_ = np.zeros([n, d, n_values])
    n_range = np.arange(n)
    for j in range(d):
        X_[n_range, :, X[:, j]] = 1
    return X_


# https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
def random_choice_prob_index(a):
    # r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    axis = len(a.shape) - 1
    r = np.expand_dims(np.random.rand(*a.shape[:-1]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def choice(a, n):
    cumsum = np.stack([a.cumsum(axis=-1)]*n)
    r = np.expand_dims(np.random.rand(n, *a.shape[:-1]), axis=-1)
    return (cumsum > r).argmax(axis=-1)


def second_order_ustat_variance_ustat(H):
    n = H.shape[0]
    di = np.diag_indices(n)
    Ht = H.copy()
    Ht[di] = 0
    sum_sq = np.sum(Ht)**2
    frob_sq = np.sum(Ht**2)
    rowsum_norm = np.sum(np.sum(Ht, axis=1)**2)
    n4 = n*(n-1)*(n-2)*(n-3)
    variance = (n+1)*rowsum_norm - (n-1)*frob_sq - sum_sq
    fac = np.array(4.*(n-2)/(n*(n-1)*n4), dtype=np.float64)
    # print(fac*(n+1)*rowsum_norm, fac*(n-1)*frob_sq, fac*sum_sq)
    variance *= fac
    v2 = (frob_sq/(n*(n-1)) - (sum_sq-4.*rowsum_norm + 2.*frob_sq)/n4)
    v2 *= 2./(n*(n-1))
    # print('{} + {} = {}'.format(variance, v2, variance+v2))
    variance += v2
    return variance


def second_order_ustat_variance_vstat(H):
    n = H.shape[0]
    A = np.sum(np.sum(H, axis=1)**2) / (n**3)
    B = np.sum(H)**2 / (n**4)
    variance = 4.*(n-2)/(n*(n-1)) * (A - B)
    # C = np.mean(H**2)
    # variance += 2./(n*(n-1)) * (C - B)
    return variance


def second_order_ustat_variance_jackknife(H):
    n = H.shape[0]
    di = np.diag_indices(n)
    Ht = H.copy()
    Ht[di] = 0.
    ustat = np.sum(Ht) / (n*(n-1))

    variance = 0.

    for i in range(n):
        idx = np.arange(n)
        idx = np.delete(idx, i)
        H_ = Ht[idx][:, idx]
        mean = np.sum(H_) / ((n-1)*(n-2))
        tmp = (mean-ustat)**2
        variance = variance + (tmp-variance) / (i+1)
    variance = (n-1)*variance

    return variance
 