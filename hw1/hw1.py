#!/usr/bin/env python

import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.special import expit


def odds_ratio(X, Y, Z, data):
    """
    Compute the odds ratio OR(X, Y | Z).
    X, Y are names of variables
    in the data frame. Z is a list of names of variables.

    Return float OR for the odds ratio OR(X, Y | Z)
    """
    if len(Z) > 0:
        design = data[[X] + Z]
    else:
        design = data[[X]]
    design = sm.tools.add_constant(design)
    beta_1 = sm.Logit(data[Y], design).fit(disp=0).params[X]
    OR = np.exp(beta_1)
    return OR


def compute_confidence_intervals(X, Y, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals through bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []

    for i in range(num_bootstraps):
        rows_i = np.random.choice(np.arange(data.shape[0]), data.shape[0])
        data_i = data.iloc[rows_i]
        estimates.append(odds_ratio(X, Y, Z, data_i))

    q_low = np.quantile(estimates, Ql)
    q_up = np.quantile(estimates, Qu)

    return q_low, q_up


def main():
    """
    Do not edit this function. This function is used for grading purposes only.
    """

    np.random.seed(200)
    data = pd.read_csv("data.txt")
    print(odds_ratio("opera", "mortality", [], data), compute_confidence_intervals("opera", "mortality", [], data))
    print(odds_ratio("opera", "mortality", ["income"], data),
          compute_confidence_intervals("opera", "mortality", ["income"], data))

    print(odds_ratio("mortality", "opera", [], data), compute_confidence_intervals("mortality", "opera", [], data))
    print(odds_ratio("mortality", "opera", ["income"], data),
          compute_confidence_intervals("mortality", "opera", ["income"], data))


if __name__ == "__main__":
    # main()
    np.random.seed(200)
    data = pd.read_csv("data.txt")
    print(odds_ratio("opera", "mortality", ["income", "college_degree"], data),
          compute_confidence_intervals("opera", "mortality", ["income", "college_degree"], data))

