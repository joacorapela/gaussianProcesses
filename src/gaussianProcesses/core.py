
import pdb
import math
import numpy as np

class GPMarginalLogLikelihood(object):
    def __init__(self, x, y, kernel):
        self._x = x
        self._y = y
        self._kernel = kernel

        self._kyInv = None
        self._logdetKy = None
        self._params = None

    def _computeKyInvAndLogDet(self, params):
       ky = self._kernel.buildKSample(x1=self._x, x2=self._x, params=params)
       self._kyInv = np.linalg.inv(a=ky)
       (_, self._logdetKy) = np.linalg.slogdet(a=ky)
       self._params = params

    def evalWithoutInverse(self, params):
        # K_y^-1 * y = u
        # y = K_y * u
        # y.T * K_y^-1 * y = y.T * u

        n = len(self._y)
        ky = self._kernel.buildKSample(x1=self._x, x2=self._x, params=params)
        u = np.linalg.solve(a=ky, b=self._y)
        answer = -0.5*np.dot(a=self._y, b=u)
        (_, logdetKy) = np.linalg.slogdet(a=ky)
        answer = answer - 0.5*logdetKy - n/2* math.log(2*math.pi)

        return answer

    def eval(self, params):
        if not np.array_equal(a1=params, a2=self._params):
            self._computeKyInvAndLogDet(params=params)
        n = len(self._y)
        answer = -.5*np.dot(a=np.dot(a=self._y, b=self._kyInv), b=self._y)\
                 -.5*self._logdetKy\
                 -n/2*math.log(2*math.pi)
        return answer

    def evalGradient(self, params):
        if not np.array_equal(a1=params, a2=self._params):
            self._computeKyInvAndLogDet(params=params)

        alpha = np.dot(a=self._kyInv, b=self._y)
        kyGrad = self._kernel.buildKSampleGrad(x1=self._x, x2=self._x, params=params)
        gradValues = np.empty(len(params))
        matrixConst = np.outer(a=alpha, b=alpha)-self._kyInv
        for i in range(len(kyGrad.keys())):
            gradValues[i] = .5*np.trace(np.matmul(matrixConst, 
                                                   kyGrad.values[i]))
        answer = dictionary(keys=kyGrad.keys(), values=gradValues)
        return grad

    def evalWithGradient(self, params):
        value = self.eval(params=params)
        grad = self.evalGradient(params=params)
        answer = {}
        answer['value'] = value
        answer['grad'] = grad
        return answer
