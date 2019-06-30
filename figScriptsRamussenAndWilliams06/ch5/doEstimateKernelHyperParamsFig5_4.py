
import sys
import pdb
import numpy as np
from scipy.optimize import minimize
import os
srcGaussianProcessesDirname = os.path.join(os.path.dirname(__file__),
                                            '../../src/gaussianProcesses')
sys.path.append(srcGaussianProcessesDirname)
from core import GPMarginalLogLikelihood
from kernels import SquaredExponentialKernel

def main(argv):
    dataFilenamePattern = "../ch2/results/dataFig2_5_l%.2f_sf%.2f_sn%.2f.npz"
    resultFilenamePattern = "results/optimResFig2_5_l%.2f_sf%.2f_sn%.2fl0%.2fsf0%.02fsn0%.02f.npz"

    if len(argv)!=7:
        raise ValueError("%s requires six arguments: <trueL> <trueSf> <trueSn> <l0> <sf0> <sn0>"%(argv[0]))

    trueL = float(argv[1])
    trueSf = float(argv[2])
    trueSn = float(argv[3])
    l0 = float(argv[4])
    sf0 = float(argv[5])
    sn0 = float(argv[6])
    minimizeMethod = 'BFGS'

    dataFilename = dataFilenamePattern%(trueL, trueSf, trueSn)
    resultFilename = resultFilenamePattern%(trueL, trueSf, trueSn, l0, sf0, sn0)
    loadRes = np.load(file=dataFilename)
    x = loadRes["x"]
    y = loadRes["y"]
    x0 = np.array([l0, sf0, sn0])

    kSquaredExponential = SquaredExponentialKernel()
    ml = GPMarginalLogLikelihood(x=x, y=y, kernel=kSquaredExponential)
    evaluationPoints = [np.array([l0,sf0,sn0])]

    def callbackFun(xk):
        print("evaluation at (%.4f, %.4f)"%(xk[0], xk[1]))
        evaluationPoints.append(xk)

    def evalWrapper(params, sign=1.0):
        return sign*ml.eval(params=params)

    def evalGradientWrapper(params, sign=1.0):
        return sign*ml.evalGradient(params=params)

    res = minimize(fun=evalWrapper, x0=x0, method=minimizeMethod, 
                                jac=evalGradientWrapper, args=(-1.0),
                                callback=callbackFun, options={'disp': True})
    np.savez(file=resultFilename, fun=res.fun, x=res.x, evaluationPoints=np.array(evaluationPoints))
    print("optimum at l=%.04f, sf=%.04f, sn=%.04f"%(res.x[0], res.x[1], res.x[2]))
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
