import numpy as np
from scipy import interpolate
import pdb

def _estim_dist_old(quantiles, percentiles, y_min, y_max, smooth_tails, tau):
    """ Estimate CDF from list of quantiles, with smoothing """

    noise = np.random.uniform(low=0.0, high=1e-8, size=((len(quantiles),)))
    noise_monotone = np.sort(noise)
    quantiles = quantiles + noise_monotone

    # Smooth tails
    def interp1d(x, y, a, b):
        return interpolate.interp1d(x, y, bounds_error=False, fill_value=(a, b), assume_sorted=True)

    cdf = interp1d(quantiles, percentiles, 0.0, 1.0)
    inv_cdf = interp1d(percentiles, quantiles, y_min, y_max)

    if smooth_tails:
        # Uniform smoothing of tails
        quantiles_smooth = quantiles
        tau_lo = tau
        tau_hi = 1-tau
        q_lo = inv_cdf(tau_lo)
        q_hi = inv_cdf(tau_hi)
        idx_lo = np.where(percentiles < tau_lo)[0]
        idx_hi = np.where(percentiles > tau_hi)[0]
        if len(idx_lo) > 0:
            quantiles_smooth[idx_lo] = np.linspace(quantiles[0], q_lo, num=len(idx_lo))
        if len(idx_hi) > 0:
            quantiles_smooth[idx_hi] = np.linspace(q_hi, quantiles[-1], num=len(idx_hi))

        cdf = interp1d(quantiles_smooth, percentiles, 0.0, 1.0)
        inv_cdf = interp1d(percentiles, quantiles_smooth, y_min, y_max)

    return cdf, inv_cdf

def _estim_dist(quantiles, percentiles, y_min, y_max, smooth_tails, tau):
    """ Estimate CDF from list of quantiles, with smoothing """

    noise = np.random.uniform(low=0.0, high=1e-5, size=((len(quantiles),)))
    noise_monotone = np.sort(noise)
    quantiles = quantiles + noise_monotone

    # Smooth tails
    def interp1d(x, y, a, b):
        return interpolate.interp1d(x, y, bounds_error=False, fill_value=(a, b), assume_sorted=True)

    cdf = interp1d(quantiles, percentiles, 0.0, 1.0)
    inv_cdf = interp1d(percentiles, quantiles, y_min, y_max)

    if smooth_tails:
        # Uniform smoothing of tails
        quantiles_smooth = quantiles
        tau_lo = tau
        tau_hi = 1-tau
        q_lo = inv_cdf(tau_lo)
        q_hi = inv_cdf(tau_hi)
        idx_lo = np.where(percentiles < tau_lo)[0]
        idx_hi = np.where(percentiles > tau_hi)[0]
        if len(idx_lo) > 0:
            quantiles_smooth[idx_lo] = np.linspace(quantiles[0], q_lo, num=len(idx_lo))
        if len(idx_hi) > 0:
            quantiles_smooth[idx_hi] = np.linspace(q_hi, quantiles[-1], num=len(idx_hi))

        cdf = interp1d(quantiles_smooth, percentiles, 0.0, 1.0)
        inv_cdf = interp1d(percentiles, quantiles_smooth, y_min, y_max)

    # Standardize
    breaks = np.linspace(y_min, y_max, num=1000, endpoint=True)
    cdf_hat = cdf(breaks)
    f_hat = np.diff(cdf_hat)
    f_hat = (f_hat+1e-6) / (np.sum(f_hat+1e-6))
    cdf_hat = np.concatenate([[0],np.cumsum(f_hat)])
    cdf = interp1d(breaks, cdf_hat, 0.0, 1.0)
    inv_cdf = interp1d(cdf_hat, breaks, y_min, y_max)

    return cdf, inv_cdf


class Histogram():
    def __init__(self, percentiles, breaks):
        self.percentiles = percentiles
        self.breaks = breaks

    def compute_histogram(self, quantiles, ymin, ymax, alpha, smooth_tails=True):
        """
        Compute pi_hat[j]: the mass between break[j-1] and break[j]
        """
        n = quantiles.shape[0]
        B = len(self.breaks)-1

        pi_hat = np.zeros((n,B+1))
        percentiles = np.concatenate(([0],self.percentiles,[1]))
        quantiles = np.pad(quantiles, ((0,0),(1, 1)), 'constant', constant_values=(ymin,ymax))

        def interp1d(x, y, a, b):
            return interpolate.interp1d(x, y, bounds_error=False, fill_value=(a, b), assume_sorted=True)

        for i in range(n):
            cdf, inv_cdf = _estim_dist(quantiles[i], percentiles, y_min=ymin, y_max=ymax, 
                                       smooth_tails=smooth_tails, tau=0.01)
            cdf_hat = cdf(self.breaks)
            pi_hat[i] = np.concatenate([[0], np.diff(cdf_hat)])
            pi_hat[i] = (pi_hat[i]+1e-6) / (np.sum(pi_hat[i]+1e-6))

        return pi_hat
