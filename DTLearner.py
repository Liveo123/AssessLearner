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

        Each row in our table will consist of:
        Node no Factor  SplitVal Left   Right
        3       11      3.4      4      7

        Here:
        Node No:    The line in the table
        Factor:     The column no we are using for the split (best corr)
        SpliVal:    The median of the column we are using to split
        Left:       Row in this table which starts the left tree
        Right:      Row in this table which starts the right tree
        '''

        if self.verbose:
            mu.printVerbose("dataX", dataX)
            mu.printVerbose("dataY", dataY)
        
        # If there is only one row, it's a leaf, so send it back.
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
            
            if self.verbose:
                mu.printVerbose("Coef Matrix", coef_matrix)
            
            # Need to grab all of those items in the coefficient Matrix which
            # make up the correlation values needed. Those are all the of the
            # numbers which correlate each row in dataX with dataY.
            correl = [np.abs(coef_matrix[coef_matrix.shape[1] - 1, i]) for i in range(0, coef_matrix.shape[1] - 1)]

            if self.verbose:
                mu.printVerbose("Correl Values", correl)

            # Find the place of the maximum correlation and its position in the 
            # matrix.
            max_pos = np.argmax(correl)
            max_corr = correl[max_pos]  
            
            # Split all of the data according to the median value of the 
            # correlation and add to its own table.  N.B.  The splits go
            # through the whole table, grabbing each column that is less 
            # and adding it to the left table and grabbing each that is 
            # bigger and adding it to the right table.
            split_val = np.median(dataX[:, max_pos])
            
            split_left = [dataX[:, max_pos] <= split_val]
            mu.printVerbose("dataX[split_left]", dataX[split_left])

            split_right = [dataX[:, max_pos] > split_val]
            mu.printVerbose("dataX[split_right]", dataX[split_right])
           
            # ???
            if np.sum(split_left) == dataX[:, max_pos].shape[0]:
                return np.array([-1, np.mean(dataY), -1, -1])

            # Use the split up tables to create their own subtree on both
            # the left side and the right using recursion.
            left_tree = self.buildTree(dataX[split_left], dataY[split_left])
            right_tree = self.buildTree(dataX[split_right], dataY[split_right])

            # Create the root node.  This is found by giving it:
            # max_pos - This is best corr col - i.e. the one we use for split.
            # split_val - The value in the corr col we split on (median)
            # The start of the left tree (always next in table)
            # The start of the right tree (half way down).
            root = [max_pos, split_val, 1, left_tree.shape[0] + 1]

            # Return the root along with either side of the tree.
            # We concatencate these all together in the table, in
            # the order, root, left tree then right tree
            return(append(root, left_tree, right_tree))

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
