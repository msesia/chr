import numpy as np
from scipy.stats import norm
import scipy.special as sp
from operator import mul
from functools import reduce
import pdb
from scipy.stats import multivariate_normal
from scipy.stats.mstats import mquantiles

class Model_Ex1:
    def __init__(self, a=1.0, symmetry=0):
        self.a = a
        self.symmetry = symmetry

    def sample_X(self, n):
        X = np.random.uniform(0, self.a, size=n)
        X = X.reshape((n,1))
        return X.astype(np.float32)

    def _sample_Y(self, x):
        y = np.random.poisson(np.sin(x*2*np.pi)**2+0.1) + 0.2*x*np.random.randn(1)
        y += (np.random.uniform(0,1,1)<0.09)*(5+2*np.random.randn(1))
        # Toss a coin and decide whether to flip y
        if np.random.uniform(0,1,1)<self.symmetry:
            y = -y
        return y
    
    def sample_Y(self, X):
        Y = 0*X
        for i in range(len(X)):
            Y[i] = self._sample_Y(X[i])
        return Y.astype(np.float32).flatten()

    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y
        
class Model_Ex2:
    def __init__(self, a=1.0, symmetry=0):
        self.a = a
        self.symmetry = symmetry

    def sample_X(self, n):
        X = np.random.uniform(0, self.a, size=n)
        X = X.reshape((n,1))
        return X.astype(np.float32)

    def _sample_Y(self, x):
        y = np.random.poisson(np.sin(x*2*np.pi)**2+0.1) + 1*x*np.random.randn(1)
        y += (np.random.uniform(0,1,1)<0.09)*(5+2*np.random.randn(1))
        # Toss a coin and decide whether to flip y
        if np.random.uniform(0,1,1)<self.symmetry:
            y = -y
        return y
    
    def sample_Y(self, X):
        Y = 0*X
        for i in range(len(X)):
            Y[i] = self._sample_Y(X[i])
        return Y.astype(np.float32).flatten()

    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y


class Model_Ex3:
    def __init__(self, p=1):
        self.p = p
        self.beta = np.zeros((self.p,))
        self.beta[0:5] = 1.0

    def sample_X(self, n):
        X = np.random.uniform(size=(n,self.p))
        return X.astype(np.float32)

    def sample_Y(self, X):
        n = X.shape[0]
        def f(Z):
            return(2.0*np.sin(np.pi*Z) + np.pi*Z)

        Z = np.dot(X,self.beta)
        E = np.random.normal(size=(n,))
        Y = f(Z) + np.sqrt(1.0+Z**2) * E

        return Y.astype(np.float32).flatten()

    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y

class Model_Ex4:
    def __init__(self, a=0.9):
        self.a = a

    def sample_X(self, n):
        X = np.random.uniform(0.1, self.a, size=n)
        X = X.reshape((n,1))
        return X.astype(np.float32)

    def sample_Y(self, X):
        Y = 0*X
        for i in range(len(X)):
            Y[i] = np.sin(X[i]*np.pi) + X[i]*np.random.randn(1)
            
        return Y.astype(np.float32).flatten()

    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y


def covariance_AR1(p, rho):
    """
    Construct the covariance matrix of a Gaussian AR(1) process
    """
    assert len(rho)>0, "The list of coupling parameters must have non-zero length"
    assert 0 <= max(rho) <= 1, "The coupling parameters must be between 0 and 1"
    assert 0 <= min(rho) <= 1, "The coupling parameters must be between 0 and 1"

    # Construct the covariance matrix
    Sigma = np.zeros(shape=(p,p))
    for i in range(p):
        for j in range(i,p):
            Sigma[i][j] = reduce(mul, [rho[l] for l in range(i,j)], 1)
    Sigma = np.triu(Sigma)+np.triu(Sigma).T-np.diag(np.diag(Sigma))
    return Sigma
    
class Model_GaussianAR1:
    """
    Gaussian AR(1) model
    """
    def __init__(self, p=10, rho=0.7):
        """
        Constructor
        :param p      : Number of variables
        :param rho    : A coupling parameter
        :return:
        """
        self.p = p
        self.rho = rho
        self.Sigma = covariance_AR1(self.p, [self.rho]*(self.p-1))
        self.mu = np.zeros((self.p,))

    def sample(self, n=1, **args):
        """
        Sample the observations from their marginal distribution
        :param n: The number of observations to be sampled (default 1)
        :return: numpy matrix (n x p)
        """
        return np.random.multivariate_normal(self.mu, self.Sigma, n).astype(np.float)
    
    def extract_y(self, X, feature_y=0):
        Y = X[:,feature_y]
        X = np.delete(X, feature_y, 1)
        return X.astype(np.float), Y.astype(np.float)
    
    def flip_signs(self,X,percent_flip=0.001):
        is_outlier = np.random.binomial(1, percent_flip, X.shape)
        new_X = (1-2*is_outlier)*X
            
        return new_X.astype(np.float), is_outlier
    
    def p_not_outlier(self,X,feature_y,percent_flip=0.001):
        dist = multivariate_normal(mean=self.mu, cov=self.Sigma)
        X_tilde = X.copy()
        X_tilde[:,feature_y] = -X_tilde[:,feature_y]
        fx = dist.pdf(X)
        fx_tilde = dist.pdf(X_tilde)
        ret_val = 1.0 - fx_tilde*percent_flip / (fx_tilde*percent_flip + fx*(1-percent_flip))
        return ret_val

        
