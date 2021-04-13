#!/usr/bin/env python

import pandas as pd
import numpy as np
from fcit import fcit


def nonparametric_fcit_test(X, Y, Z, data):
    """
    X and Y are names of variables.
    Z is a list of names of variables.
    data is a pandas data frame.

    Return a float corresponding to the p-value computed from FCIT.
    """
    x_np = np.transpose(np.asmatrix(data[X].values))
    y_np = np.transpose(np.asmatrix(data[Y].values))
    z_np = np.asmatrix(data[Z].values)

    p = fcit.test(x_np, y_np, z_np)
    return p

def main():
    """
    Do not edit this function. This function is used for grading purposes only.
    """

    np.random.seed(0)
    data = pd.read_csv("data.txt")

    print(nonparametric_fcit_test("raf", "erk", ["mek"], data))
    print(nonparametric_fcit_test("raf", "erk", ["mek", "pka", "pkc"], data))

if __name__ == "__main__":
    main()
