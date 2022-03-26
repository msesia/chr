# Conformal histogram regression

This repository contains a Python implementation of the conformal prediction method described in the accompanying paper: https://arxiv.org/abs/2105.08747

```
Matteo Sesia, Yaniv Romano
"Conformal Prediction using Conditional Histograms"
NeurIPS 2021 (spotlight)
```

## Overview

Conformal histogram regression (CHR) computes prediction intervals for non-parametric
regression that can automatically adapt to skewed data. It leverages black-box
machine learning algorithms to estimate the conditional distribution of the outcome
using histograms, and then translates their output into the shortest prediction
intervals with approximate conditional coverage. 
The theoretical results presented in the paper prove the resulting prediction
intervals have marginal coverage in finite samples, while asymptotically achiev-
ing conditional coverage and optimal length if the black-box model is consistent.
The code included here was utilized to carry out the numerical experiments with simulated and real data described in the paper,
which demonstrate empirically the improved performance of CHR compared to state-of-the-art alternatives, including conformalized quantile
regression and other distributional conformal prediction approaches.

## Dependencies

This code is written for Python (v 3.7.6) and makes use of the following packages:
 - torch (v 1.6.0)
 - numpy (v 1.18.5)
 - pandas (v 1.1.0)
 - scipy (v 1.4.1)
 - six (v 1.15.0)
 - sklearn (v 0.23.2)
 - skgarden (v 0.1.2)
 - rpy2 (v 3.3.5) (optional and requires R installation with BART package, not used in the paper)

The tutorial notebook in the "examples/" directory is written for Jupyter and was originally compiled with the following setup.
 - jupyter core     : 4.6.3
 - jupyter-notebook : 6.0.3
 - qtconsole        : not installed
 - ipython          : 7.13.0
 - ipykernel        : 5.2.0
 - jupyter client   : 6.1.2
 - jupyter lab      : not installed
 - nbconvert        : 5.6.1
 - ipywidgets       : 7.6.3
 - nbformat         : 5.0.4
 - traitlets        : 4.3.3 
 
 This repository also included code to process the experimental results and produce the figures shown in the paper.
 This is written in [R](https://www.r-project.org/) (v 4.0.3) and relies on the [tidyverse](https://www.tidyverse.org/) package (v 1.3.0).
 
 ## Instructions

Our method is implemented in the package contained within the "chr/" directory.
This can be loaded and utilized as demonstrated in the tutorial notebook "examples/intro.ypynb".

The Python code needed to reproduce our numerical experiments are in the "experiments/" directory,
along with bash scripts to submit the experiments, either sequentially (default), or on a computing cluster with a slurm interface.
The script "experiments/dataset.py" loads and pre-processes the real data sets, which can be dowloaded freely from the sources referenced in the accompanying paper. 
