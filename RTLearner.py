"""
A simple wrapper for Random Decision Trees.  (c) 2017 Paul Livesey
"""

import numpy as np
import myutils as mu

class RTLearner(object):

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
            # Split the data on a random on a random feature
            rnd_pos = np.random.randint(0, (dataX.shape[1]-1)) 

            # Also split on a random number (instead of median as in basic DT)
            rnd_split_val1 = np.random.randint(0, (dataX.shape[0]-1))
            rnd_split_val2 = np.random.randint(0, (dataX.shape[0]-1))

            # Split all of the data according to the median value of the 
            # correlation and add to its own table.  N.B.  The splits go
            # through the whole table, grabbing each column that is less 
            # and adding it to the left table and grabbing each that is 
            # bigger and adding it to the right table.
            mu.printVerbose("rnd_pos", rnd_pos)
            mu.printVerbose("rnd_split_val_1", rnd_split_val1)
            mu.printVerbose("rnd_split_val_2", rnd_split_val2)

            split_val = (dataX[rnd_split_val1, rnd_pos] + dataX[rnd_split_val2, rnd_pos]) / 2
            
            split_left = [dataX[:, rnd_pos] <= split_val]
            mu.printVerbose("dataX[split_left]", dataX[split_left])

            split_right = [dataX[:, rnd_pos] > split_val]
            mu.printVerbose("dataX[split_right]", dataX[split_right])
           
            # ???
            if np.sum(split_left) == dataX[:, rnd_pos].shape[0]:
                return np.array([-1, np.mean(dataY), -1, -1])

            # Use the split up tables to create their own subtree on both
            # the left side and the right using recursion.
            left_tree = self.buildTree(dataX[split_left], dataY[split_left])
            mu.printVerbose("Left Tree", left_tree)
            right_tree = self.buildTree(dataX[split_right], dataY[split_right])
            mu.printVerbose("Right Tree", right_tree)

            if len(left_tree.shape) == 1:
                num_on_left = 2
            else:
                num_on_left = left_tree.shape[0] + 1

            # Create the root node.  This is found by giving it:
            # rnd_pos - This is best corr col - i.e. the one we use for split.
            # split_val - The value in the corr col we split on (median)
            # The start of the left tree (always next in table)
            # The start of the right tree (half way down).
            root = [rnd_pos, split_val, 1, num_on_left]

            # Return the root along with either side of the tree.
            # We concatencate these all together in the table, in
            # the order, root, left tree then right tree
            return(np.vstack((root, left_tree, right_tree)))

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
        @param points: should be a numpy array with each row corresponding to a 
        specific query.
        @returns the estimated values according to the saved model.
        """
        return (self.model_coefs[:-1] * points).sum(axis = 1) \
                + self.model_coefs[-1]

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"