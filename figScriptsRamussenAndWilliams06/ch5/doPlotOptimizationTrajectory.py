
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os
srcGaussianProcessesDirname = os.path.join(os.path.dirname(__file__),
                                            '../../src/gaussianProcesses')
sys.path.append(srcGaussianProcessesDirname)
from core import GPMarginalLogLikelihood
from kernels import SquaredExponentialKernel

def main(argv):
    dataFilenamePattern = '../ch2/results/dataFig2_5_l%.2f_sf%.2f_sn%.2f.npz'
    optimResFilenamePattern = 'results/optimResFig2_5_l%.2f_sf%.2f_sn%.2fl0%.2fsf0%.02fsn0%.02f.npz'
    figFilenamePattern = 'figures/optimTrajectory_l%.2f_sf%.2f_sn%.2fl0%.2fsf0%.2fsn0%.2f.png'

    if len(argv)!=8:
        raise ValueError('%s requires eight arguments: <nro sample points per axis> <trueL> <trueSf> <trueSn> <l0> <sf0> <sn0>'%(argv[0]))

    snStart = -2.5
    snStop = 0.5
    lStart = -1.0
    lStop = 2.0
    nLevels = 30
    tooSmallThr = -60
    annotateColor = 'red'
    xlabel = 'characteristic lengthscale'
    ylabel = 'noise standard deviation'

    nSamplePointsPerAxis = int(argv[1])
    trueL = float(argv[2])
    trueSf = float(argv[3])
    trueSn = float(argv[4])
    l0 = float(argv[5])
    sf0 = float(argv[6])
    sn0 = float(argv[7])

    nSnPoints = nSamplePointsPerAxis
    nLPoints = nSamplePointsPerAxis
    dataFilename = dataFilenamePattern%(trueL, trueSf, trueSn)
    loadRes = np.load(file=dataFilename)
    x = loadRes['x']
    y = loadRes['y']

    optimResFilename = optimResFilenamePattern%(trueL, trueSf, trueSn, l0, sf0, sn0)
    loadRes = np.load(file=optimResFilename)
    optimizationTrajectory = loadRes['evaluationPoints']

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
            params = np.array([l, sf, sn])
            mls[i, j] = ml.eval(params=params)
    tooSmallIndices = np.where(mls<tooSmallThr)
    mls[tooSmallIndices] = tooSmallThr

    x = ls
    y = sns
    [X, Y] = np.meshgrid(x, y)
    Z = mls
    cp = plt.contour(X, Y, Z, nLevels)
    plt.plot([trueL], [trueSn], marker='*', color='red')
    plt.plot(abs(optimizationTrajectory[:,0]),
              abs(optimizationTrajectory[:,2]), 
              color="red", marker="o", linestyle="-")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    figFilename = figFilenamePattern%(trueL, trueSf, trueSn, l0, sf0, sn0)
    plt.savefig(fname=figFilename)

    plt.show()

    pdb.set_trace()

if __name__=='__main__':
    main(sys.argv)
