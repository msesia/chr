import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as n2r
#from rpy2.rinterface_lib.embedded import RRuntimeError # raising error
#import rpy2.robjects.packages as rpackagess

#from rpy2.robjects.packages import importr
#utils = importr('utils')
#utils.install_packages('quantregForest')
#utils.install_packages('BART')
# devtools::install_github("rizbicki/FlexCoDE")
# devtools::install_github("rizbicki/predictionBands")

import numpy as np
from scipy.stats.mstats import mquantiles

import pdb 

# Activate R interface
n2r.activate()

class DistSplit:
    """
    Classical CQR
    """
    def __init__(self):
        self.r = ro.r
        self.r.library('FlexCoDE')
        self.r.library('predictionBands')
      
    def fit_calibrate(self, X, Y, alpha, bbox=None, random_state=2020, verbose=False):
        self.alpha = alpha
        self.fit = self.r['fit_predictionBands'](X, Y)

    def predict(self, X):
        n = X.shape[0]

        pred_r = np.array(self.r['predict'](self.fit, X, type="dist", **{'alpha':self.alpha}))
        pred = np.zeros((n,2))
        for i in range(n):
            pred_str = pred_r[4][i][0].replace(',', ' ').replace('(', '').replace(')', '').split()
            pred[i,0] = float(pred_str[0])
            pred[i,1] = float(pred_str[1])

        return pred
