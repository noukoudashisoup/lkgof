# all utility functions in kgof.util are visible.
from kgof.util import *
import numpy as np
import autograd.numpy as anp
from scipy import sparse


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
    """Sampling from an array of probability vectors. 
    Where the last dimension of the array represents the object index over which
    the probability is defined.
    If a' shape is (d_1, ..., d_{K-1}, d_K), then the probability is over
    d_K objects.  

    Args:
        a (ndarray): probability tensor. a.sum(axis=d_K) should be all one. 

    Returns:
        Sample from a: ndarray of size (d_1, ...,d_{K-1}).
    """
    # r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    axis = len(a.shape) - 1
    r = np.expand_dims(np.random.rand(*a.shape[:-1]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def random_choice(a, n):
    """Sampling n samples from an array of probability vectors. 
    Where the last dimension of the array represents the object index over which
    the probability is defined.
    If a' shape is (d_1, ..., d_{K-1}, d_K), then the probability is over
    d_K objects.  

    Args:
        a (ndarray): probability tensor. a.sum(axis=d_K) should be all one. 

    Returns:
        Sample from a: ndarray of size (n, d_1, ...,d_{K-1}).
        where each element takes values in {0, ..., d_K-1}.
    """
    # r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    axis = len(a.shape) - 1
    r = np.expand_dims(np.random.rand(*a.shape[:-1], n), axis=axis)
    cumsum = np.expand_dims(a.cumsum(axis=axis), axis=-1)
    idx = [axis] + [i for i in range(0, axis)]
    return (r < cumsum).argmax(axis=axis).transpose(idx)


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

def featurize_bow(X, W):
    # vocab size
    n, d = X.shape
    data = np.ones(n*d)
    indices = X.ravel()
    indptr = np.arange(0, n*d+1, d)
    X_ = sparse.csr_matrix((data, indices, indptr), shape=(n, W))
    return X_

def rejection_sampling(n, cond_fn, sample_fn, seed=30):
    with NumpySeedContext(seed=seed):
        n_accept = 0 
        samples = []
        while n_accept < n:
            sample = sample_fn()
            idx = cond_fn(sample)
            samples.append(sample[idx])
            n_accept += np.count_nonzero(idx)
        return np.vstack(samples)[:n]


def softplus(x, beta=1, threshold=20):
    idx = (x*beta > threshold)
    y = np.empty_like(x)
    y = np.log1p(np.exp(beta*x))/beta
    y[idx] = x[idx]
    return y

def inv_softplus(x, beta=1, threshold=20):
    if (x < 0).any():
        raise ValueError("input has to be positive")
    y = np.log(np.expm1(beta*x))/beta
    return y

def test_sample():
    N = 3
    K = 5
    T = 10
    p = np.zeros([N,K])
    p[:N, :N] = np.eye(N)
    samples = []
    for i in range(T):
        samples.append(random_choice_prob_index(p))
    sample = np.array(samples)
    sample_ = random_choice(p, T)
    print(sample)
    print(sample_)

def test_rejection_sampler():
    n = 10
    d = 5
    sampler_fn = lambda : np.random.randn(n, d)
    cond_fn = lambda X: (X>=0).all(axis=1)
    X = rejection_sampling(n, cond_fn, sampler_fn)
    print(X.shape)

def smooth_rect(X):
    idx = (X<=1e-2)
    Y = anp.where(idx, 1e-2, X)
    return anp.where(idx, 1e-15, anp.exp(-1/Y))

def der_smooth_rect(X):
    idx = (X>1e-2)
    return anp.where(idx, anp.divide(anp.exp(-1/X), X**2), 1e-15)

def cutoff(X, a, b):
    Fb = smooth_rect(b-X)
    Fa = smooth_rect(X-a)
    logdiff = anp.log(Fa)-anp.log(Fb)
    idx = anp.logical_or(Fb < 1e-12, logdiff>25)
    Y = anp.where(idx, 1e-15, 1./(1.+anp.exp(logdiff)))
    idx_ = (Fa < 1e-12)
    return anp.where(idx_, 1., Y)

def der_cutoff(x, a, b):
    Fb = smooth_rect(b-x)
    Fa = smooth_rect(x-a)
    d1 = -der_smooth_rect(b-x) / (Fb+Fa)
    d2 = -cutoff(x, a, b)/(Fb + Fa) * (der_smooth_rect(x-a)-der_smooth_rect(b-x))
    return d1 + d2

def bump_l2(X, r, frac=0.95):
    if frac <= 0 or frac >1:
        raise ValueError('frac has to between 0 and 1')
    norm = anp.sqrt(anp.sum(X**2, axis=-1)+1e-15)
    return cutoff(norm, frac*r, r)

def partial_bump_l2(X, r, dim, frac=0.95):
    """Return partial derivative
    with respect to dim-th coordinate of 
    (n,)-array"""
    if frac <= 0 or frac >1:
        raise ValueError('frac has to between 0 and 1')
    _, d = X.shape
    Xd = X[:, dim]
    norm = anp.sum(X**2, axis=-1)**0.5
    r_ = frac * r
    if d == 1:
        return der_cutoff(norm, r_, r)
    return der_cutoff(norm, r_, r) * Xd / norm


def grad_bump_l2(X, r, frac=0.95):
    """Return gradient of bump function
    of (n,d)-array"""
    if frac <= 0 or frac >1:
        raise ValueError('frac has to between 0 and 1')
    norm = anp.sum(X**2, axis=-1, keepdims=True)**0.5
    r_ = frac * r
    P = der_cutoff(norm, r_, r) * X / norm
    return P

def ball_sampler(n, d, radius):
    X = np.random.randn(n, d) 
    Xnorm = np.sum(X**2, axis=1, keepdims=True)**0.5
    u = np.random.rand(n)
    r = radius * u**(1./d)
    return r * X/Xnorm


def main():
    n = 10
    d = 3
    X = np.random.randn(n, d)
    meds = dimwise_meddistance(X,)
    for i in range(d):
        print(meddistance(X[:, i, np.newaxis]))
    print(meds)

if __name__ == '__main__':
    # test_sample()
    test_rejection_sampler()
    # main()
