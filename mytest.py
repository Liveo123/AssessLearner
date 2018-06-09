import numpy as np
import DTLearner as dt

learner = dt.DTLearner(leaf_size = 1, verbose = True)

Xtrain = np.array([[0,1,2,4],[0,1,4,0],[0,1,6,0],[0,1,8,4]])
Ytrain = np.array([0,0,0,0])

learner.addEvidence(Xtrain, Ytrain)


