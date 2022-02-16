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
    Htsum = np.sum(Ht)

    lfn3 = lower_factorial(n, 3)
    variance = 0.
    for i in range(n):
        tmp = 4.*( (n*Ht[i, :].sum() - Htsum) / lfn3)**2
        # sequentially compute the empirical mean
        variance = variance + (tmp-variance) / (i+1)
    variance = (n-1)*variance

    return variance
 

def dimwise_dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0] x X.shape[1]
    """
    sx = X**2
    sy = Y**2
    D2 =  sx[:, np.newaxis, :] + sy[np.newaxis, :, :] - 2.0*np.einsum('ij,kj->ikj', X, Y)
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def dimwise_meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise, dimension wise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and 
        there are more slightly more 0 than 1. In this case, the m

    Return
    ------
    array of median distance of size d
    """
    if subsample is None:
        D = dimwise_dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri].reshape(-1, X.shape[-1])
        med = np.median(Tri, axis=0)
        if np.any(med) <= 0:
            # use the mean
            return np.mean(Tri, axis=0)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return dimwise_meddistance(X[ind, :], None, mean_on_fail)


def lower_factorial(n, k):
    """Returns \prod_{i=0}^{k-1} (n-i) """
    if k == 0:
        return 1.
    return np.prod([(n-ik) for ik in range(k)])


def main():
    n = 10
    d = 3
    X = np.random.randn(n, d)
    meds = dimwise_meddistance(X,)
    for i in range(d):
        print(meddistance(X[:, i, np.newaxis]))
    print(meds)

if __name__ == '__main__':
    main()
