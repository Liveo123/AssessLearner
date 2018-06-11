"""
A simple wrapper for Decision Trees.  (c) 2017 Paul Livesey
"""

import numpy as np
import myutils as mu

class DTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'plivesey3' # replace tb34 with your Georgia Tech username

    def buildTree(self, dataX, dataY):
        ''' 
        @summary: Recursive algorithm to build decision tree. 
        (Based on alrgorithm by JR Quinlin)
        @param dataX: X values of data to add
        @param dataY: the Y training values 
        '''
        if self.verbose:
            mu.printVerbose("dataX", dataX)
            mu.printVerbose("dataY", dataY)
        
        # if data.shape[0]==1:return[leaf, data.y, NA, NA]
        if dataX.shape[0] == 1:
            return np.array([-1, dataY[-1], -1, -1])
        # if all data.y same: return[leaf, data.y, NA, NA]
        # This is done with list comprehension.  Create a list of all the data
        # that is the same as the first element.  If that list if the size of
        # all of the original data, they are all the same, so return a leaf.
        elif len([sameData for sameData in dataY[1:dataY.shape[0]] \
            if sameData == dataY[0]]) == dataY.shape[0] - 1:

            return np.array([-1, dataY[0], -1, -1])
        # else
        else:
            # Determine best feature max_corr to split on
            # i.e. Go through all of columns in dataX and find the one which
            # corrleates most with dataY (the results column)
            coef_matrix = np.corrcoef(dataX, dataY, rowvar=False)
            correl = [np.abs(coef_matrix[coef_matrix.shape[1] - 1, i]) for i in range(0, coef_matrix.shape[1] - 1)]

            if self.verbose:
                mu.printVerbose("Correl Values", correl)

            max_pos = np.argmax(correl)
            max_corr = correl[max_pos]  
            
            SplitVal = np.median(dataX[:, max_pos])
            split_left = [dataX[: i] <= SplitVal]
            split_right = [dataX[:, i] > SplitVal]
            mu.printVerbose("dataX[split_left]", dataX[split_left])
            mu.printVerbose("dataX[split_right]", dataX[split_right])
            lefttree = self.buildTree(dataX[split_left], dataY[split_left])
            righttree = self.buildTree(dataX[split_right], dataY[split_right])
            root = [i, SplitVal, 1, lefttree.shape[0] + 1]
            return(append(root, lefttree, righttree))

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # build and save the model
        new_tree = self.buildTree(dataX, dataY)
        if self.verbose:
            print(new_tree)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
