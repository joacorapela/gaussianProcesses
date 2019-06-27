
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
    minX = -10
    maxX = 10
    xlab= "x"
    ylab = "y"
    marker = "x"
    ci95PAlpha = .5
    ci95PColor = "gray"
    dataFilenamePattern = "results/dataFig2_5_l%.2f_sf%.2f_sn%.2f.npz"
    figFilenamePattern = "figures/dataFig2_5_trueL%.2f_trueSf%.2f_trueSn%.2f_predL%.2f_predSf%.2f_predSn%.2f.png"

    if len(argv)!=8:
        raise ValueError("%s requires seven arguments: <number of points per GP sample> <trueL> <trueSf> <trueSn> <predL> <predSf> <predSn>"%(argv[0]))

    mu = lambda x: 0
    nStar = int(argv[1])
    trueL = float(argv[2])
    trueSf = float(argv[3])
    trueSn = float(argv[4])
    predL = float(argv[5])
    predSf = float(argv[6])
    predSn = float(argv[7])
    dataFilename = dataFilenamePattern%(trueL, trueSf, trueSn)
    loadRes = np.load(file=dataFilename)
    x = loadRes["x"]
    y = loadRes["y"]
    n = len(x)
    xStar = np.sort(np.random.uniform(low=minX, high=maxX, size=nStar))
    params = {"sf": predSf, "l":predL, "sn":predSn}
    kSquaredExponential = SquaredExponentialKernel()
    kXX = kSquaredExponential.buildKSample(x1=x, x2=x, params=params)
    kXStarX = kSquaredExponential.buildKSample(x1=xStar, x2=x, params=params)
    kXStarXStar = kSquaredExponential.buildKSample(x1=xStar, x2=xStar,
                                                             params=params)
    v = np.linalg.solve(a=kXX+(predSn**2)*np.identity(n=n), b=y)
    muPosterior = np.dot(a=kXStarX, b=v)
    u = np.linalg.solve(a=kXX+(predSn**2)*np.identity(n=n), b=kXStarX.T)
    kPosterior = kXStarXStar-np.dot(a=kXStarX, b=u)
    posteriorSTDSample = np.sqrt(np.diag(kPosterior))
    ci95Pup = muPosterior+2*posteriorSTDSample
    ci95Pdown = muPosterior-2*posteriorSTDSample

    plt.scatter(x, y, marker=marker)
    plt.plot(xStar, muPosterior)
    plt.fill_between(xStar, ci95Pdown, ci95Pup, alpha=ci95PAlpha, color=ci95PColor)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    figFilename = figFilenamePattern%(trueL, trueSf, trueSn, predL, predSf, predSn)
    plt.savefig(fname=figFilename)
    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

