import numpy as np
import RTLearner as dt
import myutils as mu

learner = dt.RTLearner(leaf_size = 1, verbose = True)

Xtrain = np.array([[0.885,0.330,9.100],
                   [0.725,0.390,10.900],
                   [0.560,0.500,9.400],
                   [0.735,0.570,9.800],
                   [0.610,0.630,8.400],
                   [0.260,0.630,11.800],
                   [0.500,0.680,10.500],
                   [0.320,0.780,10.000]])
                            
Ytrain = np.array([4.000,5.000,6.000,5.000,3.000,8.000,7.000,6.000]).T

#if learner.verbose == True:
#print(np.c_[Xtrain, Ytrain])    
    #mu.printVerbose(np.c_[Xtrain, Ytrain])    

print(learner.addEvidence(Xtrain, Ytrain))


