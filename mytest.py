import numpy as np
import DTLearner as dt
import myutils as mu

learner = dt.DTLearner(leaf_size = 1, verbose = True)

Xtrain = np.array([[5,1,5,2],
                   [6,5,4,4],
                   [0,3,2,1],
                   [0,2,6,6],
                   [4,5,1,1],
                   [0,3,3,4]])
                            
Ytrain = np.array([1,2,3,4,5,5]).T

#if learner.verbose == True:
#print(np.c_[Xtrain, Ytrain])    
    #mu.printVerbose(np.c_[Xtrain, Ytrain])    

learner.addEvidence(Xtrain, Ytrain)


