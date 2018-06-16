import numpy as np
import DTLearner as dt
import BagLearner as bl
import LinRegLearner as ln
import myutils as mu

learner = bl.BagLearner(dt.DTLearner, kwargs={"leaf_size": 1}, bags = 15, boost = False, verbose = True)
#learner = bl.BagLearner(ln.LinRegLearner, kwargs={}, bags = 10, boost = False, verbose = True)

Xtrain = np.array([[0.885,0.330,9.100],
                   [0.725,0.390,10.900],
                   [0.560,0.500,9.400],
                   [0.735,0.570,9.800],
                   [0.535,0.270,7.800],
                   [0.135,0.670,3.800],
                   [0.610,0.630,8.400],
                   [0.260,0.630,11.800],
                   [0.500,0.680,10.500],
                   [0.320,0.780,10.000]])

Ytrain = np.array([4.000,5.000,6.000,5.000,3.000,8.000,7.000,6.000]).T

print(learner.addEvidence(Xtrain, Ytrain))
                            

Xtest = np.array([[0.665,0.440, 8.400],
                  [0.776,0.444, 6.555],
                  [0.376,0.844, 3.555],
                  [0.554,0.333, 9.000]])

res = learner.query(Xtest)


print("=== result ===")
print('\n{}\n'.format(res))
