
import pytest
import pandas as pd

def test_k():
    tol = 1e-6
    dataFilename = "results/squaredExponentialKernel_testData.csv"
    df = pandas.from_csv(dataFilename)
    kernel = SquaredExponentialKernel()

    for i in range(df.shape[0]):
        x1 = df[i, "x1"]
        x2 = df[i, "x2"]
        l  = df[i, "l"]
        sf  = df[i, "sf"]
        sn  = df[i, "sn"]
        k  = df[i, "k"]
        params = {"l": l, "sf": sf, "sn": sn}
        newK = kernel._k(x1=x1, x2=x2, params=params)
        if math.abs(newK-k)>tol: return 1
    return 0
