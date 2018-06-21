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

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape
    
    # Create stores for the results
    in_corr = np.empty(100)
    in_rmse = np.empty(100)
    out_corr = np.empty(100)
    out_rmse = np.empty(100)

    # test leaf sizes of 1 to 100
    for cnt in range (0, LEAF_SIZE_RANGE):

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
    plt.ylabel('RMSE')
    plt.xlabel('Leaf Size')
    plt.title('RMSE DTLearner redwine')
    plt.ylim(0.4, 0.9)
    plt.show()


    # Corralation Graph
    blue_line = mlines.Line2D([], [], color='blue', markersize=15, label='Out-Sample')
    red_line = mlines.Line2D([], [], color='red', markersize=15, label='In-Sample')
    plt.legend(handles=[blue_line, red_line])
    plt.plot(in_corr, 'r', out_corr, 'b')
    plt.grid = True
    plt.ylabel('Correlation')
    plt.xlabel('Leaf Size')
    plt.title('Correlation DTLearner redwine')
    plt.ylim(0.6, 0.8)
    plt.show()





