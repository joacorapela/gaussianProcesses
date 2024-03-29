
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
        self._lastParams = None

    def _computeKyInvAndLogDet(self, params):
       ky = self._kernel.buildKSample(x1=self._x, x2=self._x, params=params)
       self._kyInv = np.linalg.inv(a=ky)
       (_, self._logdetKy) = np.linalg.slogdet(a=ky)
       self._lastParams = params

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
        self._computeKyInvAndLogDet(params=params)
        n = len(self._y)
        answer = -.5*np.dot(a=np.dot(a=self._y, b=self._kyInv), b=self._y)\
                 -.5*self._logdetKy\
                 -n/2*math.log(2*math.pi)
        return answer

    def evalGradient(self, params):
        if (self._lastParams is None) or (not np.array_equal(params, self._lastParams)):
            self._computeKyInvAndLogDet(params=params)

        alpha = np.dot(a=self._kyInv, b=self._y)
        kyGrad = self._kernel.buildKSampleGrad(x1=self._x, x2=self._x, params=params)
        # kyGradValues = list(kyGrad.values())
        # gradValues = np.empty(len(params))
        grad = np.empty(len(params))
        matrixConst = np.outer(a=alpha, b=alpha)-self._kyInv
        for i in range(len(kyGrad)):
            grad[i] = .5*np.trace(np.matmul(matrixConst, kyGrad[i]))
        return grad

    def evalWithGradient(self, params):
        value = self.eval(params=params)
        grad = self.evalGradient(params=params)
        answer = {}
        answer['value'] = value
        answer['grad'] = grad
        return answer
