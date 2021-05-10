import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as n2r

import numpy as np
from scipy.stats.mstats import mquantiles

# Activate R interface
n2r.activate()

class QRF:
    """ Quantile random forests (implemented in R)
    """
    def __init__(self, quantiles, n_estimators=100, min_samples_leaf=5, 
                 n_jobs=1, random_state=0, verbose=False):
        """ Initialization
        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        n_jobs: number of parallel jobs (default: 1)
        random_state : integer, seed used in CV when splitting to train-test
        """

        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.nthreads = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        self.r = ro.r
        self.r.library('quantregForest')

    def fit(self, X, Y, return_loss=None):
        self.r('set.seed')(self.random_state)
    
        self.forest = self.r['quantregForest'](X, Y, ntree=self.n_estimators, nodesize=self.min_samples_leaf,
                                               nthreads=self.nthreads, **{'do.trace':self.verbose})


    def predict(self, X):
        """Estimate the label given the features
        Parameters
        ----------
        x : numpy array of training features (nXp)
        Returns
        -------
        ret_val : numpy array of predicted labels (n)
        """

        # Sample from posterior distribution learnt by BART
        pred = np.array(self.r['predict'](self.forest, X, what=self.quantiles))
        
        return pred
    
    def get_quantiles(self):
        return self.quantiles
    
class QBART:
    """ Bayesian Additive Regression Trees (implemented in R)
    """
    def __init__(self, quantiles, random_state=0):
        """ Initialization
        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        random_state : integer, seed used in CV when splitting to train-test
        """

        self.quantiles = quantiles

        self.r = ro.r
        self.r.library('BART')

    def fit(self, X, Y, return_loss=None):
        self.posterior = self.r['gbart'](X, Y, nskip=200, ndpost=1000)


    def predict(self, X):
        """Estimate the label given the features
        Parameters
        ----------
        x : numpy array of training features (nXp)
        Returns
        -------
        ret_val : numpy array of predicted labels (n)
        """

        # Sample from posterior distribution learnt by BART
        post_samples = np.array(self.r['predict'](self.posterior, X))
        pred = mquantiles(post_samples, prob=self.quantiles, axis=0).T

        return pred

    def get_quantiles(self):
        return self.quantiles
