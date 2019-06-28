
import sys
import pdb
import numpy as np
import pandas as pd
sys.path.append("../")
from kernels import SquaredExponentialKernel

def generateData_SquaredExponentialKernel(dataFilename):
    l = 1
    sf = 1
    sn = .1

    params = {"l": l, "sf": sf, "sn": sn}
    x = np.array([-8.0, -6.0, -5.5, -5.1, -4.5,  -4.1, -2.0, -1.5, -1.0, 0.3,  0.8,   1.0, 2.5,  2.6,  4.2,  4.25, 4.9,  6.0,  6.5])

    arraySize = len(x)**2
    ls = np.empty(arraySize)
    ls[:] = np.nan
    sfs = np.empty(arraySize)
    sfs[:] = np.nan
    sns = np.empty(arraySize)
    sns[:] = np.nan
    kernel = SquaredExponentialKernel()
    x1s = np.empty(arraySize)
    x1s[:] = np.nan
    x2s = np.empty(arraySize)
    x2s[:] = np.nan
    ks = np.empty(arraySize)
    ks[:] = np.nan
    dls = np.empty(arraySize)
    dls[:] = np.nan
    dsfs = np.empty(arraySize)
    dsfs[:] = np.nan
    dsns = np.empty(arraySize)
    dsns[:] = np.nan
    n = 0
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(x)):
            x2 = x[j]
            k = kernel._k(x1=x1, x2=x2, params=params)
            dk = kernel._kGrad(x1=x1, x2=x2, params=params)
            dl = dk["dl"]
            dsf = dk["dsf"]
            dsn = dk["dsn"]

            ls[n] = l
            sfs[n] = sf
            sns[n] = sn
            x1s[n] = x1
            x2s[n] = x2
            ks[n] = k
            dls[n] = dl
            dsfs[n] = dsf
            dsns[n] = dsn

            n += 1

    data = {"l": ls, "sf": sfs, "sn": sns, "x1": x1s, "x2": x2s, "k": ks, "dl": dls, "dsf": dsfs, "dsn": dsns}
    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=dataFilename, index=False)

    pdb.set_trace()

def main(argv):
    dataFilename = "results/squaredExponentialKernel_testData.csv"
    generateData_SquaredExponentialKernel(dataFilename=dataFilename)

if __name__=="__main__":
    main(sys.argv)
