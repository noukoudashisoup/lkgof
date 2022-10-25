"""Module containing kernel-related classes"""

from kgof.kernel import *
import lkgof.util as util
import autograd.numpy as np
from lkgof.density import DiscreteFunction
from lkgof.util import der_smooth_rect, featurize_bow, smooth_rect
import scipy.spatial.distance
import autograd


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


class KSTProduct(KSTKernel):
    """KSTKernel representing the product of two kernel

    Attributes:
        k1:
            KSTKernel object
        k2: 
            KSTKernel object
    """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def eval(self, X, Y):
        K = self.k1.eval(X, Y)
        K *= self.k2.eval(X, Y)
        return K

    def pair_eval(self, X, Y):
        K = self.k1.pair_eval(X, Y)
        K *= self.k2.pair_eval(X, Y)
        return K

    def gradX_Y(self, X, Y, dim):
        k1 = self.k1
        k2 = self.k2 
        K1 = k1.eval(X, Y)
        K2 = k2.eval(X, Y)
        G1 = k1.gradX_Y(X, Y, dim)
        G2 = k2.gradX_Y(X, Y, dim)
        return K1 * G2 + K2 * G1

    def gradY_X(self, X, Y, dim):
        return self.gradX_Y(X, Y, dim).T

    def gradX(self, X, Y):
        k1 = self.k1
        k2 = self.k2 
        K1 = k1.eval(X, Y)
        K2 = k2.eval(X, Y)
        G1 = k1.gradX(X, Y)
        G2 = k2.gradX(X, Y)
        T1 = np.einsum('ij,ijk->ijk', K1, G2)
        T2 = np.einsum('ij,ijk->ijk', K2, G1)
        return T1 + T2

    def gradXY_sum(self, X, Y):
        d = X.shape[1]
        assert Y.shape[1] == d
        k1 = self.k1
        k2 = self.k2
        K1 = k1.eval(X, Y)
        K2 = k2.eval(X, Y)
        T1 = K1 * k2.gradXY_sum(X, Y) + K2*k1.gradXY_sum(X, Y)
        T2 = np.zeros_like(T1)
        for j in range(d):
            T2 += k1.gradX_Y(X, Y, j) * k2.gradY_X(X, Y, j)
            T2 += k2.gradX_Y(X, Y, j) * k1.gradY_X(X, Y, j)
        return T1 + T2

# end of KSTProduct

class WeightFunction:
    """
    Abstract class for a feature map of a kernel.
    Assume the map is differentiable.
    """

    @abstractmethod
    def __call__(self, x):
        """
        Return a feature vector for the input x.
        """
        raise NotImplementedError()

    @abstractmethod
    def parX(self, X, dim):
        """Return the partial derivative at X w.r.t. dim"""
        pass
    
    def grad(self, X):
        d = X.shape[1]
        return np.concatenate([self.parX(X, dim)[:, np.newaxis] for dim in range(d)], axis=1)
        # return autograd.elementwise_grad(self)(X)


# end of WeightFunction

class KSTWeight(KSTKernel):
    """KSTKernel representing the product of two kernels
    Attributes:
        w_func: WeightFunction
    """
    def __init__(self, w_func):
        self.w_func = w_func

    def _eval(self, X, Y):
        w = self.w_func
        Wx = w(X)
        Wy = w(Y)
        return Wx, Wy

    def eval(self, X, Y):
        Wx, Wy = self._eval(X, Y)
        return np.outer(Wx, Wy)

    def pair_eval(self, X, Y):
        Wx, Wy = self._eval(X, Y)
        return Wx * Wy

    def gradX_Y(self, X, Y, dim):
        w = self.w_func
        Gx = w.parX(X, dim)
        Wy = w(Y)
        return np.outer(Gx, Wy)

    def gradY_X(self, X, Y, dim):
        return self.gradX_Y(Y, X, dim).T

    def gradXY_sum(self, X, Y):
        w = self.w_func
        Gx = w.grad(X)
        Gy = w.grad(Y)
        return Gx @ Gy.T

# End of KSTWeight

class WeightCoordinateProduct(WeightFunction):

    def __call__(self, x):
        """
        Return a feature vector for the input x.
        """
        d = x.shape[1]
        y = x / (1+x**2)**0.5
        return np.exp(np.log(y).mean(axis=1))

    def parX(self, X, dim):
        """Return the partial derivative at X w.r.t. dim"""
        _, d = X.shape
        idx = np.arange(d)
        idx = np.delete(idx, dim)
        X_ = X[:, idx]
        X_ = X_/ (1+X_**2)**0.5
        coef = np.exp(np.log(X_).sum(axis=1)/d)

        Xd = X[:, dim]
        P = (Xd**0.5 / (1+Xd**0.5))**(1./d-1) / d
        P *= 1./(1+Xd**2)**0.5 - Xd**2/(1+Xd**2)**1.5 
        return coef*P
    

class WeightIntevalProduct(WeightFunction):

    def __init__(self, L, U):
        self.L = L
        self.U = U 

    def __call__(self, x):
        """
        Return a feature vector for the input x.
        """
        d = x.shape[1]
        y = -(x-self.L) * (x-self.U) / (1+x**2)
        return np.prod(y, axis=-1)**(1./d)

    def parX(self, X, dim):
        """Return the partial derivative at X w.r.t. dim"""
        U = self.U
        L = self.L
        _, d = X.shape
        idx = np.arange(d)
        idx = np.delete(idx, dim)
        X_ = X[:, idx]
        y = -(X_-self.L[idx]) * (X_-self.U[idx]) / (1+X_**2)
        coef = np.prod(y, axis=-1)**(1./d)

        Xd = X[:, dim, ]
        norm = (1+Xd**2)
        P = (-(Xd-U[dim]) * (Xd-L[dim])/norm)**(1./d-1) / d
        P *= -(2*Xd-(U[dim]+L[dim]))/norm + 2*Xd*(Xd-U[dim])*(Xd-L[dim])/norm**2
        return coef*P
    
    def grad(self, X):
        d = X.shape[1]
        return np.concatenate([self.parX(X, dim)[:, np.newaxis] for dim in range(d)], axis=1)


class WeightSmoothIntevalProduct(WeightFunction):

    def __init__(self, L, U):
        self.L = L
        self.U = U 
        ratio = 0.99
        self.lmin = ratio * np.min(np.abs(L)) 
        self.umin = ratio * np.min(np.abs(U))

    def __call__(self, x):
        """
        Return a feature vector for the input x.
        """
        d = x.shape[1]
        y = util.cutoff(x, self.umin, np.abs(self.U))
        y += util.cutoff(-x, self.lmin, np.abs(self.L)) - 1.
        return np.prod(y+1e-15, axis=-1) ** (1./d)

    def parX(self, X, dim):
        """Return the partial derivative at X w.r.t. dim"""
        _, d = X.shape
        coef = 1.
        if d > 1:
            idx = np.arange(d)
            idx = np.delete(idx, dim)
            X_ = X[:, idx]
            U_ = self.U[idx]
            L_ = self.L[idx]
            y = util.cutoff(X_, self.umin, np.abs(U_))
            y += util.cutoff(-X_, self.lmin, np.abs(L_)) - 1
            coef = np.prod(y, axis=-1) ** (1./d)

        Xd = X[:, dim, ]
        P = util.cutoff(Xd, self.umin, np.abs(self.U[dim]))
        P += util.cutoff(-Xd, self.lmin, np.abs(self.L[dim])) - 1.
        P = (P+1e-15)**(1./d-1) / d
        P1 = util.der_cutoff(Xd, self.umin, np.abs(self.U[dim]))
        P2 = util.der_cutoff(-Xd, self.lmin, np.abs(self.L[dim]))
        return coef*(P1+P2)*P


class ExpTruncWeight(WeightFunction):

    def __call__(self, x):
        """
        Return a feature vector for the input x.
        """
        idx = (x > 0)
        y = np.zeros_like(x)
        y[idx] = np.exp(-1/x[idx])
        return np.prod(y, axis=-1)

    def parX(self, X, dim):
        """Return the partial derivative at X w.r.t. dim"""
        n, d = X.shape
        idx = np.arange(d)
        idx = np.delete(idx, dim)
        X_ = X[:, idx]
        coef = self(X_)

        Xd = X[:, dim, np.newaxis]
        P = (self(Xd)/Xd[:, 0]**2).reshape([n])
        return coef*P
    
    def grad(self, X):
        d = X.shape[1]
        return np.concatenate([self.parX(X, dim)[:, np.newaxis] for dim in range(d)], axis=1)



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
        return self.gradX_Y(Y, X, dim).T
        
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
    """Class representing Bag of Words kernels.
    BoW vectors are normalized by the number of words in a document. 

    Attribtues:
        n_values: 
            lattice sizes
        d:
            dimensionality 
    """
    
    def __init__(self, n_values, d):
        super(KBoW, self).__init__(n_values, d)
    
    def featurize(self, X):
        W = self.n_values[0]
        return featurize_bow(X, W)

    def eval(self, X, Y):
        d = self.dim()
        X_ = self.featurize(X)
        Y_ = self.featurize(Y)
        K = X_.dot(Y_.T) / d**2
        return K.toarray()
    
    def pair_eval(self, X, Y):
        d = self.dim()
        X_ = self.featurize(X)
        Y_ = self.featurize(Y)
        K = X_.multiply(Y_).sum(axis=1) / d**2
        return np.array(K)


    def gradX_Y(self, X, Y, dim, shift=-1):
        """
        Default: compute the (cyclic) backward difference with respect to 
        the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        d = X.shape[1]
        fY = self.featurize(Y)
        feat_Xd = self.featurize(X[:, [dim]])
        feat_Xd_ = self.featurize(np.mod(X[:, [dim]]+shift, self.n_values[dim]))
        return (feat_Xd - feat_Xd_).dot(fY.T) / d**2

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
        featurize = self.featurize

        X_shift = np.mod(X+shift, n_values)
        Y_shift = np.mod(Y+shift, n_values)
        fY = featurize(Y)
        for j in range(d):
            Y_ = Y.copy()
            Y_[:, j] = np.mod(Y[:, j]+shift, self.n_values[j])
            fY_s = featurize(Y_)

            feat_Xd = featurize(X[:, [j]])
            feat_Xd_s = featurize(X_shift[:, [j]])

            K += (feat_Xd - feat_Xd_s).dot(fY.T) / d**2
            K += (feat_Xd_s - feat_Xd).dot(fY_s.T) /d**2
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


class KGaussBoW(DKSTKernel):
    """Class representing the Gaussian kernel defined on
    Bag-of-Words vectors.

    Attribtues:
        n_values: 
            lattice sizes
        d:
            dimensionality
    """
 
    def __init__(self, n_values, d, s2=1.0):
        super(KGaussBoW, self).__init__(n_values, d)
        self.kbow = KBoW(n_values, d)
        self.s2 = s2
    
    def eval(self, X, Y):
        kbow = self.kbow
        eval = kbow.eval
        pair_eval = kbow.pair_eval
        s2 = self.s2
        K = np.exp(eval(X, Y))
        Kx = np.exp(pair_eval(X, X))
        Ky = np.exp(pair_eval(Y, Y))
        return (K / (Kx * Ky.T)**0.5)**(1./s2)
    
    def pair_eval(self, X, Y):
        inner = self.kbow.pair_eval
        s2 = self.s2
        K = np.exp(inner(X, Y))
        Kx = np.exp(inner(X, X))
        Ky = np.exp(inner(Y, Y))
        return (K / (Kx * Ky)**0.5)**(1./s2)


class KIMQBoW(DKSTKernel):
    """Class representing the Gaussian kernel defined on
    Bag-of-Words vectors.

    Attribtues:
        n_values: 
            lattice sizes
        d:
            dimensionality
        c (float): a bias parameter
        b (float): exponenent (b < 0)
        s2: squared length scale parameter. 
            Defaults to the squared dimension. 
    """
 
    def __init__(self, n_values, d, c=1.0, b=-0.5, s2=None):
        super(KIMQBoW, self).__init__(n_values, d)
        if not b < 0:
            raise ValueError('The exponent has to be negative')
        if not c > 0:
            raise ValueError('c has to be positive. Was {}'.format(c))
        self.b = b
        self.c = c
        self.s2 = s2 if not (s2 is None) else d**2

    def _load_params(self):
        return self.b, self.c, self.s2

    def eval(self, X, Y):
        W = self.n_values[0]
        X_ = featurize_bow(X, W)
        Y_ = featurize_bow(Y, W)
        b, c, s2 = self._load_params()

        X2 = X_.multiply(X_).sum(axis=1) 
        Y2 = Y_.multiply(Y_).sum(axis=1) 
        D2 = np.array(X2 + Y2.T - 2.*(X_.dot(Y_.T)))
        return (c**2+D2/s2)**(b)
    
    def pair_eval(self, X, Y):
        W = self.n_values[0]
        X_ = featurize_bow(X, W)
        Y_ = featurize_bow(Y, W)
        b, c, s2 = self._load_params()

        X2 = X_.multiply(X_).sum(axis=1)
        Y2 = Y_.multiply(Y_).sum(axis=1)
        D2 = np.array(X2 + Y2.T - 2.*np.array(X_.multiply(Y_).sum(axis=1)))
        return (c**2+D2/s2)**(b)


class KExpBoW(DKSTKernel):
    """Class representing the exponentiated dot-product kernel defined on
    Bag-of-Words vectors.

    Attribtues:
        n_values: 
            lattice sizes
        d:
            dimensionality
    """
 
    def __init__(self, n_values, d, s2=1.0):
        super(KExpBoW, self).__init__(n_values, d)
        self.kbow = KBoW(n_values, d)
        self.s2 = s2
    
    def eval(self, X, Y):
        kbow = self.kbow
        eval = kbow.eval
        pair_eval = kbow.pair_eval
        s2 = self.s2
        K = np.exp(eval(X, Y))
        return K**(1./s2)
    
    def pair_eval(self, X, Y):
        inner = self.kbow.pair_eval
        s2 = self.s2
        K = np.exp(inner(X, Y))
        return K**(1./s2)


class KPIMQ(KSTKernel):
    
    def __init__(self, P, c=1.0, b=-0.5):
        """Precontioned IMQ kernel
        k(x,y) = (c^2 + <(x-y), P^{-1}(x-y)>)^b
        Note that the input has to have a compatible dimension with P. 

        Args:
            c (float): a bias parameter
            b (float): exponenent (-1 < b < 0)
            P (ndarray): preconditioning matrix. Required to be positive definite.
        """
        self.c = c
        if not b < 0:
            raise ValueError('The exponent has to be negative')
        if not c > 0:
            raise ValueError('c has to be positive. Was {}'.format(c))
        self.b = b
        s, U = np.linalg.eigh((P+P.T)/2)
        if np.min(s) <= 1e-12:
            raise ValueError('P has to be positive definite')
        self.invsqrtP = U @ np.diag(s**(-0.5)) @ U.T
    
    def _load_params(self):
        return self.c, self.b, self.invsqrtP

    def eval(self, X, Y):
        """Evalute the kernel on data X and Y """
        c, b, invsqrtP = self._load_params()
        X_ = np.dot(X, invsqrtP.T)
        Y_ = np.dot(Y, invsqrtP.T)
        D2 = util.dist2_matrix(X_, Y_)
        K = (c**2 + D2)**b
        return K
    
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...
        """
        assert X.shape[0] == Y.shape[0]
        c, b, invsqrtP = self._load_params()
        X_ = np.dot(X, invsqrtP.T)
        Y_ = np.dot(Y, invsqrtP.T)

        return (c**2 + np.sum((X_-Y_)**2, 1))**b

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        c, b, invsqrtP = self._load_params()
        X_ = np.dot(X, invsqrtP.T)
        Y_ = np.dot(Y, invsqrtP.T)
        D2 = util.dist2_matrix(X_, Y_)
        diff = (X_[:, np.newaxis] - Y_[np.newaxis])
        p = invsqrtP[:, dim]
        Gdim = ( 2.0*b*(c**2 + D2)**(b-1) )[:, :, np.newaxis] * diff
        Gdim = np.dot(Gdim, p)
        assert Gdim.shape[0] == X.shape[0]
        assert Gdim.shape[1] == Y.shape[0]
        return Gdim

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return self.gradX_Y(Y, X, dim).T

    def gradXY_sum(self, X, Y):
        """
        Compute
        \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        c, b, invsqrtP = self._load_params()
        P_ = np.dot(invsqrtP, invsqrtP.T)
        X_ = np.dot(X, invsqrtP.T)
        Y_ = np.dot(Y, invsqrtP.T)
        diff = (X_[:, np.newaxis, :] - Y_[np.newaxis])
        D2 = util.dist2_matrix(X_, Y_)

        c2D2 = c**2 + D2
        T1 = np.einsum('ij, nmi, nmj->nm', P_, diff, diff)
        T1 = -T1 * 4.0*b*(b-1)*(c2D2**(b-2))
        T2 = -2.0*b*np.trace(P_)*c2D2**(b-1) 
        return T1 + T2

# end class KPIMQ

def main2():
    n = 100
    d = 50
    v = 1000
    n_values = np.array(d*[v])
    k = KBoW(n_values=n_values, d=d)
    X = np.random.randint(0, v, [n, d])
    from lkgof.util import ContextTimer
    with ContextTimer() as t:
        K1 = k.gradXY_sum(X, X)
        K2 = super(KBoW, k).gradXY_sum(X, X)
    print(K1-K2)
    print(t.secs)

def main():
    n = 10
    d = 5
    X = np.random.randn(n, d)
    Y = np.random.randn(n+1, d)
    P = np.eye(5)
    k = KPIMQ(P, c=1.0, b=-0.5)
    kimq = KIMQ()
    K1 = (k.gradX_Y(X, Y, 0))
    K2 = kimq.gradX_Y(X, Y, 0)

    K1 = k.gradXY_sum(X, Y)
    K2 = kimq.gradXY_sum(X, Y)
    print(np.amax(K1-K2))


if __name__ == '__main__':
    main2()
