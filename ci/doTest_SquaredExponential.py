
import pdb
import sys
sys.path.append('../src/gaussianProcesses/')
from test_SquaredExponential import test_k

def main(argv):
    rc = test_k()
    pdb.set_trace()
    return rc

if __name__=="__main__":
    main(sys.argv)
