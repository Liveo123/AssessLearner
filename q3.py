"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import pandas as pd
import BagLearner as bl
import DTLearner as dt
import RTLearner as rt
import sys
import matplotlib.pyplot as plt
import myutils as mu
import matplotlib.lines as mlines

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

    # Create stores for the results for DT
    in_corr_dt = np.empty(100)
    in_rmse_dt = np.empty(100)
    out_corr_dt = np.empty(100)
    out_rmse_dt = np.empty(100)

    # Create stores for the results for DT
    in_corr_rt = np.empty(100)
    in_rmse_rt = np.empty(100)
    out_corr_rt = np.empty(100)
    out_rmse_rt = np.empty(100)

    # test leaf sizes of 1 to 100 for the DTLearner
    for cnt in range (0, LEAF_SIZE_RANGE):

        # create a learner and train it
        learner = bl.BagLearner(dt.DTLearner, kwargs={"leaf_size":cnt+1}, bags=20, boost=False, verbose = False)
        learner.addEvidence(trainX, trainY) # train it
        print learner.author()

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        in_rmse_dt[cnt] = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", in_rmse_dt[cnt]
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]
        in_corr_dt[cnt] = c[0,1]
    #    mu.printVerbose("trainY", trainY.head())

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        out_rmse_dt[cnt] = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", out_rmse_dt[cnt]
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]
        out_corr_dt[cnt] = c[0,1]

    #    mu.printVerbose("trainY", trainY.head())

    # test leaf sizes of 1 to 100 for the RTLearner
    for cnt in range (0, LEAF_SIZE_RANGE):

        # create a learner and train it
        learner = bl.BagLearner(rt.RTLearner, kwargs={"leaf_size":cnt+1}, bags=20, boost=False, verbose = False)
        #learner = rt.RTLearner(leaf_size=cnt+1, verbose = False)
        learner.addEvidence(trainX, trainY) # train it
        print learner.author()

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        in_rmse_rt[cnt] = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", in_rmse_rt[cnt]
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]
        in_corr_rt[cnt] = c[0,1]

    #    mu.printVerbose("trainY", trainY.head())

        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        out_rmse_rt[cnt] = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", out_rmse_rt[cnt]
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]
        out_corr_rt[cnt] = c[0,1]



        
    #plt.plot(in_corr, 'i', out_corr, 'g')
    #plt.ylim(0.6,0.95)
    #plt.show()
    #plt.plot(in_rmse, out_rmse)
    #plt.ylim(0.004, 0.009)
    #plt.show()

    blue_line = mlines.Line2D([], [], color='blue', markersize=15, label='RT RMSE')
    red_line = mlines.Line2D([], [], color='red', markersize=15, label='DT RMSE')
    plt.legend(handles=[blue_line, red_line])
    plt.plot(in_rmse_dt, 'r', in_rmse_rt, 'b')
    plt.grid = True
    plt.ylabel('RMSE')
    plt.xlabel('Leaf Size')
    plt.title('RMSE DTLearner vs RTLearner Istanbul')
    plt.ylim(0.000, 0.1)
    plt.show()

    # Corralation Graph
    blue_line = mlines.Line2D([], [], color='blue', markersize=15, label='RT Correlation')
    red_line = mlines.Line2D([], [], color='red', markersize=15, label='DT Correlation')
    plt.legend(handles=[blue_line, red_line])
    plt.plot(in_corr_dt, 'r', out_corr_rt, 'b')
    plt.grid = True
    plt.ylabel('Correlation')
    plt.xlabel('Leaf Size')
    plt.title('Correlation DTLearner vs RTLearner Istanbul')
    plt.ylim(0.0, 0.8)
    plt.show()
