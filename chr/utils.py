import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections

from chr import coverage

import pdb

class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).float()

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

def evaluate_predictions(pred, Y, X=None):
    # Extract lower and upper prediction bands
    pred_l = np.min(pred,1)
    pred_h = np.max(pred,1)
    # Marginal coverage
    cover = (Y>=pred_l)*(Y<=pred_h)
    marg_coverage = np.mean(cover)
    if X is None:
        wsc_coverage = None
    else:
        # Estimated conditional coverage (worse-case slab)
        wsc_coverage = coverage.wsc_unbiased(X, Y, pred, M=100)

    # Marginal length
    lengths = pred_h-pred_l
    length = np.mean(lengths)
    # Length conditional on coverage
    idx_cover = np.where(cover)[0]
    length_cover = np.mean([lengths for i in idx_cover])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Length': [length], 'Length cover': [length_cover]})
    return out

def plot_histogram(breaks, weights, S=None, fig=None, limits=None, i=0, colors=None, linestyles=None, xlim=None, filename=None):
    if colors is None:
        if limits is not None:
            colors = ['tab:blue'] * len(limits)
    if linestyles is None:
        if limits is not None:
            linestyles = ['-'] * len(limits)

    if fig is None:
        fig = plt.figure()
    plt.step(breaks, weights[i], where='pre', color='black')
    if S is not None:
        idx = S[i]
        z = np.zeros(len(breaks),)
        z[idx] = weights[i,idx]
        plt.fill_between(breaks, z, step="pre", alpha=0.4, color='gray')
    if limits is not None:
        for q_idx in range(len(limits[i])):
            q = limits[i][q_idx]
            plt.axvline(q, 0, 1, linestyle=linestyles[q_idx], color=colors[q_idx])

    plt.xlabel('$Y$')
    plt.ylabel('Density')

    if xlim is not None:
        plt.xlim(xlim)

    if filename is not None:
        fig.set_size_inches(4.5, 3)
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    plt.show()
