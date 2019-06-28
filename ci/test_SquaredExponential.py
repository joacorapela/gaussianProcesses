
import sys
import pdb
import os
import pytest
import pandas as pd
from kernels import SquaredExponentialKernel

def test_k():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "results/squaredExponentialKernel_testData.csv")
    df = pd.read_csv(dataFilename)
    kernel = SquaredExponentialKernel()

    for i in range(df.shape[0]):
        x1 = df.loc[i, "x1"]
        x2 = df.loc[i, "x2"]
        l  = df.loc[i, "l"]
        sf  = df.loc[i, "sf"]
        sn  = df.loc[i, "sn"]
        k  = df.loc[i, "k"]
        params = {"l": l, "sf": sf, "sn": sn}
        newK = kernel._k(x1=x1, x2=x2, params=params)
        assert(abs(newK-k)<tol)
