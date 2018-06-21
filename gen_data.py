"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees

def best4DT(seed=1489683273):
    np.random.seed(seed)
    #X = np.zeros((100,2))
    #Y = np.random.random(size = (100,))*200-100
    X = np.random.randint(0, math.pi, size=(500,2))
    # Here we need to get any function that is not linear
    Y = np.tan(X[:,0] + X[:,1])
    
    return X, Y

def best4LinReg(seed=1489683273):

    np.random.seed(seed)
    #X = np.zeros((100,2))
    #Y = np.random.random(size = (100,))*200-100
    X = np.random.randint(0, math.pi, size=(500,2))
    #Here we just need a linear func!
    Y = X[:,0] + X[:,1]
    # Here's is an example of creating a Y from randomly generated
    # X with multiple columns
    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3
    return X, Y

def author():
    return 'plivesey3' #Change this to your user ID

if __name__=="__main__":
    print "they call me Tim."
