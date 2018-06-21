"""
A Bag Learner wrapper of a bag learner wrapper.  (c) 2017 Paul Livesey
"""

import numpy as np, LinRegLearner as lr, BagLearner as bl

class InsaneLearner(object):

    def __init__(self, 
                 bags = 20,
                 verbose = False):
        self.bags = bags
        self.verbose = verbose
        self.results = np.array([])

    def author(self):
        return 'plivesey3'

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
       
        self.results = {}

        # Go through all of the bags, adding bags of 20 LinRegLearners
        # This is insane!!
        for bag in range(self.bags):

            BinBLearner = bl.BagLearner(lr.LinRegLearner, {})
            BinBLearner.addEvidence(dataX, dataY)
            self.results[bag] = BinBLearner
                
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a 
        specific query.
        @returns the estimated values according to the saved model.
        """
        
        # Go through all of the results and find their mean.  This is our
        # main result
        results = np.empty(points.shape[0])
        cnt = 0.0
        for key, dec_table in self.results.items():
            results += (dec_table.query(points))
            cnt = cnt + 1.0
        
        return results / cnt

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
