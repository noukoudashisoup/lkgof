"""Module containing kernel-related classes"""

from kgof.kernel import *
import autograd.numpy as np
from lkgof.density import DiscreteFunction
import scipy.spatial.distance
from scipy import sparse


class DKSTKernel(Kernel, DiscreteFunction):
    """
    Interface specifiying methods a kernel has to implement to be used with 
    the Discrete Kernelized Stein discrepancy test of Yang et al., 2018.
    """

    def __init__(self, n_values, d):
        """Require subclasses to have n_values and d """
        self._n_values = n_values
        self._d = d
    
    def gradX_Y(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to 
        the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """

        X_ = X.copy()
        X_[:, dim] = np.mod(X_[:, dim]+shift, self.n_values[dim])
        return (self.eval(X, Y) - self.eval(X_, Y))

    def gradY_X(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to
        the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return self.gradX_Y(Y, X, dim, shift).T

    def gradXY_sum(self, X, Y, shift=-1):
        """
        Compute the trace term in the kernel function in Yang et al., 2018. 

        X: nx x d numpy array.
        Y: ny x d numpy array. 

        Return a nx x ny numpy array of the derivatives.
        """
        nx, d = X.shape
        ny, _ = Y.shape
        K = np.zeros((nx, ny))
        n_values = self.n_values
        for j in range(d):
            X_ = X.copy()
            Y_ = Y.copy()
            X_[:, j] = np.mod(X[:, j]+shift, n_values[j])
            Y_[:, j] = np.mod(Y[:, j]+shift, n_values[j])
            K += (self.eval(X, Y) + self.eval(X_, Y_)
                  - self.eval(X_, Y) - self.eval(X, Y_))
        return K

    def dim(self):
        return self._d

# end KSTKernel


class KHamming(DKSTKernel):

    def __init__(self, n_values, d):
        """
        Args:
        - n_values: a positive integer/ integer array specifying the number of possible values
          of the discrete variable. 
        - d: dimensionality of the input 
        """
        super(KHamming, self).__init__(n_values, d)

    def _normalized_hamming_dist(self, X, Y):
        nx = X.shape[0]
        ny = Y.shape[0]
        H = np.empty([nx, ny])
        for i in range(ny):
            H[:, i] = np.mean(X!=Y[i], axis=1)
        return H

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        Args: 
            X: n x d where each row represents one point
            Y: n x d
        Return: 
            a n x n numpy array.
        """
        assert X.shape[1] == Y.shape[1]
        # hamm_dist = self._normalized_hamming_dist(X, Y)
        hamm_dist = scipy.spatial.distance.cdist(X, Y, 'hamming')
        return np.exp(-hamm_dist)

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        
        X: n x d where each row represents one point
        Y: n x d
        return a 1d numpy array of length n.
        """
        assert X.shape == Y.shape
        n, d = X.shape
        H = np.zeros((n, d))
        H[X!=Y] = 1
        return np.exp(-np.mean(H, axis=1))


class KDelta(DKSTKernel):

    def __init__(self, n_values, d):
        super(KDelta, self).__init__(n_values, d)

    def eval(self, X, Y):
        nx = X.shape[0]
        ny = Y.shape[0]
        K = np.ones((nx, ny))
        for i in range(ny):
            diff = np.sum(X!=Y[i], axis=1)
            K[diff!=0, i] = 0
        return K

    def pair_eval(self, X, Y):
        K = np.ones(X.shape[0])
        diff = np.sum(X!=Y, axis=1)
        K[diff!=0] = 0
        return K


class KExpInner(DKSTKernel):
    """ Exponentiated inner product kernel
    for binary variable input.
    """

    # can be extended to non-binary
    def __init__(self, n_values, d):
        super(KExpInner, self).__init__(n_values, d)

    def eval(self, X, Y):
        _, d = X.shape
        assert X.shape[1] == Y.shape[1]
        # assuming input is 0 or 1
        X = 2 * (X-0.5)
        Y = 2 * (Y-0.5)
        return np.exp(np.dot(X, Y.T)/d)

    def pair_eval(self, X, Y):
        assert X.shape == Y.shape
        X = 2 * (X-0.5)
        Y = 2 * (Y-0.5)
        return np.exp(np.mean(X*Y, axis=1))


def bump_l2(X, r, scale=1e-2):
    """Bump function"""
    bump = np.zeros(X.shape[0])
    l2 = np.sum(X**2, axis=1)**0.5
    idx = (l2 < r)
    bump[idx] = np.exp(-scale/(r**2 - l2[idx]**2))
    return bump


def partial_bump_l2(X, r, dim, scale=1e-2):
    """Return partial derivative
    with respect to dim-th coordinate of 
    (n,)-array"""
    pard = np.zeros(X.shape[0])
    l2 = np.sum(X**2, axis=1)**0.5
    idx = (l2 < r)
    pard[idx] = -2.*X[idx, dim] / (r**2 - l2[idx]**2)**2
    pard[idx] = scale * pard[idx] * bump_l2(X[idx], r, scale)
    return pard


def grad_bump_l2(X, r, scale=1e-2):
    """Return gradient of bump function
    of (n,d)-array"""
    grad = np.zeros(X.shape)
    l2 = np.sum(X**2, axis=1)**0.5
    idx = (l2 < r)
    grad[idx] = -2.*X[idx, :] / ((r**2 - l2[idx]**2)[:, np.newaxis])**2
    grad[idx] = scale * grad[idx] * bump_l2(X[idx], r, scale)[:, np.newaxis]
    return grad


class KGaussBumpL2(KGauss):
    """
    Gaussian kernel with L2-bump function.
    Supported in a L2-sphere with radius r centered at origin.
    Args: 

        - sigma2: kernel width
        - r: radius of support
    """

    def __init__(self, sigma2, r, scale=1e-3):
        super(KGaussBumpL2, self).__init__(sigma2)
        self.r = r
        self.scale = scale

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X : n1 x d numpy array
        Y : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        r = self.r
        scale = self.scale
        K = super(KGaussBumpL2, self).eval(X, Y)
        K = K * np.outer(bump_l2(X, r, scale), bump_l2(Y, r, scale))
        return K

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        r = self.r
        scale = self.scale
        K = super(KGaussBumpL2, self).eval(X, Y)
        G = super(KGaussBumpL2, self).gradX_Y(X, Y, dim)
        bump_pard_X = partial_bump_l2(X, r, dim, scale)
        bump_X = bump_l2(X, r, scale)
        bump_Y = bump_l2(Y, r, scale)
        return (np.outer(bump_pard_X, bump_Y)*K +
                np.outer(bump_X, bump_Y) * G)

    def pair_gradX_Y(self, X, Y):
        """
        Compute the gradient with respect to X in k(X, Y), evaluated at the
        specified X and Y.

        X: n x d
        Y: n x d

        Return a numpy array of size n x d
        """
        r = self.r
        scale = self.scale
        K = super(KGaussBumpL2, self).pair_eval(X, Y)
        G = super(KGaussBumpL2, self).pair_gradX_Y(X, Y)
        bump_grad_X = grad_bump_l2(X, r, scale)
        bump_X = bump_l2(X, r, scale)
        bump_Y = bump_l2(Y, r, scale)
        return ((bump_grad_X*K)*bump_Y + G*bump_X*bump_Y)

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return self.gradX_Y(X, Y, dim).T
        
    def pair_gradY_X(self, X, Y):
        """
        Compute the gradient with respect to Y in k(X, Y), evaluated at the
        specified X and Y.

        X: n x d
        Y: n x d

        Return a numpy array of size n x d
        """
        r = self.r
        scale = self.scale
        K = super(KGaussBumpL2, self).pair_eval(X, Y)
        G = super(KGaussBumpL2, self).pair_gradX_Y(X, Y)
        bump_grad_Y = grad_bump_l2(Y, r, scale)
        bump_X = bump_l2(X, r, scale)
        bump_Y = bump_l2(Y, r, scale)
        return ((K*bump_grad_Y)*bump_X + G*bump_X*bump_Y)

    def gradXY_sum(self, X, Y):
        r"""
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        d = X.shape[1]
        K = super(KGaussBumpL2, self).eval(X, Y)
        r = self.r
        scale = self.scale
        bump_X = bump_l2(X, r, scale)
        bump_Y = bump_l2(Y, r, scale)
        grad_bump_X = grad_bump_l2(X, r, scale)
        grad_bump_Y = grad_bump_l2(Y, r, scale)
        G = super(KGaussBumpL2, self).gradXY_sum(X, Y)
        G *= np.outer(bump_X, bump_Y)
        G += np.dot(grad_bump_X, grad_bump_Y.T) * K
        for i in range(d):
            K1 = super(KGaussBumpL2, self).gradX_Y(X, Y, i)
            L2 = np.outer(bump_X, grad_bump_Y[:, i])
            G += K1*L2 + K1.T*L2.T
        return G

    def pair_gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: n x d numpy array.
        Y: n x d numpy array.

        Return a one-dimensional length-n numpy array of the derivatives.
        """
        K = super(KGaussBumpL2, self).pair_eval(X, Y)
        G = super(KGaussBumpL2, self).pair_gradXY_sum(X, Y)
        r = self.r
        scale = self.scale
        bump_X = bump_l2(X, r, scale)
        bump_Y = bump_l2(Y, r, scale)
        grad_bump_X = grad_bump_l2(X, r, scale)
        grad_bump_Y = grad_bump_l2(Y, r, scale)
        tmp = super(KGaussBumpL2, self).pair_gradY_X(X, Y)
        tmp *= grad_bump_X * bump_Y
        G += np.sum(tmp, axis=1)
        tmp = super(KGaussBumpL2, self).pair_gradX_Y(X, Y)
        tmp *= grad_bump_Y * bump_X
        G += np.sum(tmp, axis=1)
        G += np.sum(grad_bump_X*grad_bump_Y, axis=1) * K
        return G

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        r = self.r
        scale = self.scale
        KVec = super(KGaussBumpL2, self).pair_eval(X, Y)
        bump_X = bump_X(X, r, scale)
        bump_Y = bump_Y(Y, r, scale)
        return KVec * bump_X * bump_Y


class DKLinear(DKSTKernel):
    
    def __init__(n_values, d):
        super(DKLinear, DKSTKernel).__init__(n_values, d)
        
    def eval(self, X, Y):
        d = self.dim()
        return np.dot(X, Y.T) / d
    
    def pair_eval(self, X, Y):
        d = self.dim()
        return np.sum(X*Y, axis=1) / d

    
class KBoW(DKSTKernel):
    """Class representing Bag of Words kernels

    Attribtues:
        n_values: 
            lattice sizes
        d:
            dimensionality 
    """
    
    def __init__(self, n_values, d):
        super(KBoW, self).__init__(n_values, d)
    
    def featurize(self, X):
        n_values = self.n_values
        # vocab size
        W = n_values[0]
        n, d = X.shape
        data = np.ones(n*d)
        indices = X.ravel()
        indptr = np.arange(0, n*d+1, d)
        X_ = sparse.csr_matrix((data, indices, indptr), shape=(n, W))
        return X_
            
    def eval(self, X, Y):
        d = self.dim()
        X_ = self.featurize(X)
        Y_ = self.featurize(Y)
        K = X_.dot(Y_.T) / d
        return K.toarray()
    
    def pair_eval(self, X, Y):
        d = self.dim()
        X_ = self.featurize(X)
        Y_ = self.featurize(Y)
        K = X_.multiply(Y_).sum(axis=1) / d
        return np.array(K)


class KNormalizedBoW(KBoW):
    """Class representing the normalized version of
    the Bag of Words kernel.

    Attribtues:
        n_values: 
            lattice sizes
        d:
            dimensionality
    """
 
    def __init__(self, n_values, d):
        super(KNormalizedBoW, self).__init__(n_values, d)
    
    def eval(self, X, Y):
        eval = super(KNormalizedBoW, self).eval
        pair_eval = super(KNormalizedBoW, self).pair_eval
        K = eval(X, Y)
        Kx = pair_eval(X, X)
        Ky = pair_eval(Y, Y)
        return K / (Kx * Ky.T)**0.5

    def pair_eval(self, X, Y):
        inner = super(KNormalizedBoW, self).pair_eval
        K = inner(X, Y)
        Kx = inner(X, X)
        Ky = inner(Y, Y)
        return K / (Kx * Ky)**0.5


class KGaussBoW(KBoW):
    """Class representing the Gaussian kernel defined on
    Bag-of-Words vectors.

    Attribtues:
        n_values: 
            lattice sizes
        d:
            dimensionality
    """
 
    def __init__(self, n_values, d):
        super(KGaussBoW, self).__init__(n_values, d)
    
    def eval(self, X, Y):
        eval = super(KGaussBoW, self).eval
        pair_eval = super(KGaussBoW, self).pair_eval
        K = np.exp(eval(X, Y))
        Kx = np.exp(pair_eval(X, X))
        Ky = np.exp(pair_eval(Y, Y))
        return K / (Kx * Ky.T)**0.5
    
    def pair_eval(self, X, Y):
        inner = super(KNormalizedBoW, self).pair_eval
        K = np.exp(inner(X, Y))
        Kx = np.exp(inner(X, X))
        Ky = np.exp(inner(Y, Y))
        return K / (Kx * Ky)**0.5