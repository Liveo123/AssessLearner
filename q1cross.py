"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import pandas as pd
import DTLearner as lrl
import sys
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

LEAF_SIZE_RANGE = 100
CV_RANGE = 3

if __name__=="__main__":
    if len(sys.argv) !=  2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)

    # First remove the date column from the csv
    #if sys.argv[2] == 'd':
        #f = pd.read_csv(sys.argv[1])
        #keep_cols = ['ISE-TL', 'ISE-USD', 'SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']
        #new_f = f[keep_cols]
        #new_f.to_csv(sys.argv[1], index=False)


    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()[1:]])



    # Create stores for the results
    in_corr = np.empty(100)
    in_rmse = np.empty(100)
    out_corr = np.empty(100)
    out_rmse = np.empty(100)

    # test leaf sizes of 1 to 100
    for cnt in range (0, LEAF_SIZE_RANGE):

        for cv_cnt in range(CV_RANGE):

            len_cut = data.shape[0] / CV_RANGE

            # compute how much of the data is training and testing
            train_rows_start = int(len_cut*cv_cnt)
            train_rows_end = int((len_cut*cv_cnt)+(0.6*len_cut))
            test_rows_start = int((len_cut*cv_cnt)+((0.6*len_cut)+1))
            test_rows_end = int(len_cut*(cv_cnt+1))

            #test_rows_start = data.shape[0] - train_rows

            # separate out training and testing data
            trainX = data[train_rows_start:train_rows_end,0:-1]
            trainY = data[train_rows_start:train_rows_end,-1]
            testX = data[test_rows_start:test_rows_end,0:-1]
            testY = data[test_rows_start:test_rows_end,-1]

            #mu.printVerbose("testX.shape", testX.shape)
            #mu.printVerbose("trainY.shape", trainY.shape)

            # create a learner and train it
            learner = lrl.DTLearner(leaf_size=cnt+1, verbose = True) # create a LinRegLearner
            learner.addEvidence(trainX, trainY) # train it
            print learner.author()

            # evaluate in sample
            predY = learner.query(trainX) # get the predictions
            in_rmse[cnt] = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
            print
            print "In sample results"
            print "RMSE: ", in_rmse[cnt]
            c = np.corrcoef(predY, y=trainY)
            print "corr: ", c[0,1]
            in_corr[cnt] = c[0,1]

            # evaluate out of sample
            predY = learner.query(testX) # get the predictions
            out_rmse[cnt] = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            print
            print "Out of sample results"
            print "RMSE: ", out_rmse[cnt]
            c = np.corrcoef(predY, y=testY)
            print "corr: ", c[0,1]
            out_corr[cnt] = c[0,1]

    # RMSE Graph
    blue_line = mlines.Line2D([], [], color='blue', markersize=15, label='Out-Sample')
    red_line = mlines.Line2D([], [], color='red', markersize=15, label='In-Sample')
    plt.legend(handles=[blue_line, red_line])
    plt.plot(in_rmse, 'r', out_rmse, 'b')
    plt.grid = True
    #plt.ylim(0.000, 0.01)
    plt.ylabel('RMSE')
    plt.xlabel('Leaf Size')
    plt.title('RMSE DTLearner Cross Validation Istanbul')
    plt.show()


    # Corralation Graph
    blue_line = mlines.Line2D([], [], color='blue', markersize=15, label='Out-Sample')
    red_line = mlines.Line2D([], [], color='red', markersize=15, label='In-Sample')
    plt.legend(handles=[blue_line, red_line])
    plt.plot(in_corr, 'r', out_corr, 'b')
    plt.grid = True
    plt.ylabel('Correlation')
    plt.xlabel('Leaf Size')
    plt.title('Correlation DTLearner Cross Validation Istanbul')
    #plt.ylim(0.6, 0.8)
    plt.show()
