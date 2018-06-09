"""
A simple wrapper for Decision Trees.  (c) 2017 Paul Livesey
"""

import numpy as np

class DTLearner(object):

    def __init__(self, verbose = False):
        pass # move along, these aren't the drones you're looking for

    def author(self):
        return 'plivesey3' # replace tb34 with your Georgia Tech username

    def buildTree(self, dataX, dataY):
        ''' Recursive algorithm to build decision tree. 
            (Based on alrgorithm by JR Quinlin) '''
        # if data.shape[0]==1:return[leaf, data.y, NA, NA]
        # if all data.y same: return[leaf, data.y, NA, NA]
        # else
            # Determine best feature i to split on
            # SplitVal = data[:, i].median()
            # lefttree = buildTree(data[data[:, i] <= SplitVal])
            # righttree = buildTree(data[data[:, i] > SplitVal])
            # root = [i, SplitVal, 1, lefttree.shape[0] + 1]
            # return(append(root, lefttree, righttree))

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        
        # slap on 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
        newdataX[:,0:dataX.shape[1]]=dataX

        # build and save the model
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
