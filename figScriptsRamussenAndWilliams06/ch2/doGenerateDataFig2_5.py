
import sys
import pdb
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../src/gaussianProcesses/')
from kernels import SquaredExponentialKernel

def buildMuSample(x, mu):
    muSample = np.empty(len(x))
    for i in range(len(x)):
        muSample[i] = mu(x=x[i])
    return muSample

def main(argv):
    nSamples = 1
    x = np.array([-8.0, -6.0, -5.5, -5.1, -4.5,  -4.1, -2.0, -1.5, -1.0, 0.3,  0.8,   1.0, 2.5,  2.6,  4.2,  4.25, 4.9,  6.0,  6.5])
    n = len(x)
    minX = -9
    maxX = 9
    xlab= "x"
    ylab = "y"
    marker = "x"
    mu = lambda x: 0
    resultsFilenamePattern = "results/dataFig2_5_l%.2f_sf%.2f_sn%.2f.npz"

    if len(argv)!=4:
        raise ValueError("%s requires three arguments: <l> <sf> <sn>"%(argv[0]))

    l = float(argv[1])
    sf = float(argv[2])
    sn = float(argv[3])
    muSample = buildMuSample(x=x, mu=mu)
    kSquaredExponential = SquaredExponentialKernel()
    params = {"sf":sf, "l":l, "sn":sn}
    kSample = kSquaredExponential.buildKSample(x1=x, x2=x, params=params)
    y = np.random.multivariate_normal(mean=muSample, cov=kSample)
    resultsFilename = resultsFilenamePattern%(l, sf, sn)
    np.savez(file=resultsFilename, x=x, y=y)

    plt.scatter(x, y, marker=marker)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

