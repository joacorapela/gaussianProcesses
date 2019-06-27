
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from core import GPMarginalLogLikelihood
from kernels import SquaredExponentialKernel

def main(argv):
    dataFilenamePattern = "results/dataFig2_5_l%.2f_sf%.2f_sn%.2f.npz"

    if len(argv)!=5:
        raise ValueError("%s requires four arguments: <nro sample points per axis> <trueL> <trueSf> <trueSn>"%(argv[0]))

    snStart = -2.5
    snStop = 0.5
    lStart = -1.0
    lStop = 1.0
    nLevels = 30
    tooSmallThr = -60
    annotateColor = "red"
    xlabel = "characteristic lengthscale"
    ylabel = "noise standard deviation"
    figFilenamePattern = "figures/fig2_4_trueL%.2f_trueSf%.2f_trueSn%.2f.png"

    nSamplePointsPerAxis = int(argv[1])
    trueL = float(argv[2])
    trueSf = float(argv[3])
    trueSn = float(argv[4])

    nSnPoints = nSamplePointsPerAxis
    nLPoints = nSamplePointsPerAxis
    dataFilename = dataFilenamePattern%(trueL, trueSf, trueSn)
    loadRes = np.load(file=dataFilename)
    x = loadRes["x"]
    y = loadRes["y"]

    sns = np.logspace(start=snStart, stop=snStop, num=nSnPoints)
    ls = np.logspace(start=lStart, stop=lStop, num=nLPoints)

    mls = np.empty(shape=(len(sns), len(ls)))
    mls[:] = np.nan
    kSquaredExponential = SquaredExponentialKernel()
    ml = GPMarginalLogLikelihood(x=x, y=y, kernel=kSquaredExponential)
    sf = trueSf

    for i in range(len(sns)):
        sn = sns[i]
        for j in range(len(ls)):
            l = ls[j]
            params = {"l": l, "sf": sf, "sn": sn}
            mls[i, j] = ml.eval(params=params)

    tooSmallIndices = np.where(mls<tooSmallThr)
    mls[tooSmallIndices] = tooSmallThr

    x = ls
    y = sns
    [X, Y] = np.meshgrid(x, y)
    Z = mls
    cp = plt.contour(X, Y, Z, nLevels)
    plt.plot([trueL], [trueSn], marker='*', color='red')
    # plt.grid()
    # plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    figFilename = figFilenamePattern%(trueL, trueSf, trueSn)
    plt.savefig(fname=figFilename)

    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
