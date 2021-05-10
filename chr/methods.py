import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import mquantiles

from chr.histogram import Histogram
from chr.grey_boxes import HistogramAccumulator
from chr.utils import plot_histogram

from chr.utils import evaluate_predictions
from chr.others import QR_errfun

import pdb
import matplotlib.pyplot as plt

class CHR:
    """
    Histogram-based CQR (waiting for a better name)
    """
    def __init__(self, bbox=None, ymin=-1, ymax=1, y_steps=1000, delta_alpha=0.001, intervals=True, randomize=False):

        # Define discrete grid of y values for histogram estimator
        self.grid_histogram = np.linspace(ymin, ymax, num=y_steps, endpoint=True)
        self.ymin = ymin
        self.ymax = ymax

        # Should we predict intervals or sets?
        self.intervals = intervals

        # Store the black-box
        if bbox is not None:
            self.init_bbox(bbox)

        # Store desired nominal level
        self.alpha = None
        self.delta_alpha = delta_alpha

        self.randomize = randomize


    def init_bbox(self, bbox):
        # Store the black-box
        self.bbox = bbox
        grid_quantiles = self.bbox.get_quantiles()
        # Make sure the quantiles are sorted
        assert((np.diff(grid_quantiles)>=0).all())
        # Initialize conditional histogram estimator
        self.hist = Histogram(grid_quantiles, self.grid_histogram)

    def fit(self, X, Y, bbox=None):
        # Store the black-box
        if bbox is not None:
            self.init_bbox(bbox)

        # Fit black-box model
        self.bbox.fit(X.astype(np.float32), Y.astype(np.float32))

    def calibrate(self, X, Y, alpha, bbox=None, return_scores=False):
        if bbox is not None:
            self.init_bbox(bbox)

        # Store desired nominal level
        self.alpha = alpha

        # Compute predictions on calibration data
        q_calib = self.bbox.predict(X.astype(np.float32))

        # Estimate conditional histogram for calibration points
        d_calib = self.hist.compute_histogram(q_calib, self.ymin, self.ymax, alpha)

        # Initialize histogram accumulator (grey-box)
        accumulator = HistogramAccumulator(d_calib, self.grid_histogram, self.alpha, delta_alpha=self.delta_alpha)

        # Generate noise for randomization
        n2 = X.shape[0]
        if self.randomize:
            epsilon = np.random.uniform(low=0.0, high=1.0, size=n2)
        else:
            epsilon = None

        # Compute conformity scores
        if self.intervals:
            scores = accumulator.calibrate_intervals(Y.astype(np.float32), epsilon=epsilon)
        else:
            # TODO: implement this
            assert(1==2)

        # Compute upper quantile of scores
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        self.calibrated_alpha = np.round(1.0-mquantiles(scores, prob=level_adjusted)[0],4)
        
        # Print message
        print("Calibrated alpha (nominal level: {}): {:.3f}.".format(alpha, self.calibrated_alpha))

        return self.calibrated_alpha


    def fit_calibrate(self, X, Y, alpha, random_state=2020, bbox=None,
                                        verbose=False, return_scores=False):
        # Store the black-box
        if bbox is not None:
            self.init_bbox(bbox)

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)
        n2 = X_calib.shape[0]

        # Fit black-box model
        self.fit(X_train.astype(np.float32), Y_train.astype(np.float32))

        # Calibrate
        scores = self.calibrate(X_calib.astype(np.float32), Y_calib.astype(np.float32), alpha)

        # Return conformity scores
        if return_scores:
            return scores

    def predict(self, X, alpha=None):
        assert(self.alpha is not None)

        # Compute predictions on new data
        q_new = self.bbox.predict(X.astype(np.float32))

        # Estimate conditional histogram for new data points
        d_new = self.hist.compute_histogram(q_new, self.ymin, self.ymax, self.alpha)

        # Initialize histogram accumulator (grey-box)
        accumulator = HistogramAccumulator(d_new, self.grid_histogram, self.alpha, delta_alpha=self.delta_alpha)

        # Generate noise for randomization
        n = X.shape[0]
        if self.randomize:
            epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
        else:
            epsilon = None

        # Compute prediction bands
        if alpha is None:
            alpha = self.calibrated_alpha

        _, bands = accumulator.predict_intervals(alpha, epsilon=epsilon)

        return bands
