
import sys
import pdb
import numpy as np
from scipy.optimize import minimize
from core import GPMarginalLogLikelihood
from kernels import SquaredExponentialKernel

def main(argv):
    dataFilenamePattern = "results/dataFig2_5_l%.2f_sf%.2f_sn%.2f.npz"

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
    loadRes = np.load(file=dataFilename)
    x = loadRes["x"]
    y = loadRes["y"]
    x0 = np.array([l0, sf0, sn0])

    kSquaredExponential = SquaredExponentialKernel()
    ml = GPMarginalLogLikelihood(x=x, y=y, kernel=kSquaredExponential)

    def evalWrapper(paramsArray):
        params = {"l": paramsArray[0], "sf": paramsArray[1], "sn": paramsArray[2]}
        answer = -1*ml.eval(params=params)
        print "eval(l=%.04f, sf=%.04f, sn=%.04f)=%.04f"%(params["l"], params["sf"], params["sn"], answer)
        return answer

    def evalGradientWrapper(paramsArray):
        params = {"l": paramsArray[0], "sf": paramsArray[1], "sn": paramsArray[2]}
        answer = -1*ml.evalGradient(params=params)
        return answer

    res = minimize(fun=evalWrapper, x0=x0, method=minimizeMethod, 
                                    jac=evalGradientWrapper, 
                                    options={'disp': True})
    print("optimum at l=%.04f, sf=%.04f, sfn=%.04f"%(res.x[0], res.x[1], res.x[2]))
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
