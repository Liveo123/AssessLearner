#AssessLearner Report

##Does overfitting occur with respect to leaf_size? 

Consider the dataset istanbul.csv with DTLearner. Forwhich values of leaf_size does overfitting occur? Use RMSE as your metric for assessing overfitting.

To show whether overfitting occurs with respect to leaf size, two pieces of code were written.

1) The first piece of code was created to run the DTLearner algorithm over and over, 100 times against the
/data/istanbul.csv data file and for each to record the RMSE and the collation for each.

Theoretically, it should be that overfitting occurs most in-sample when the leaf size is equal to one, because
this should fit the data perfectly. Then, as the leaf size increases,we should see the leaf the error rate
increasing.

These can then be compared with a similar plot with the out-sample data and we should see that the RMSE
starts high, but then drops down to an optimal position before moving back up again.

The results of these were then plotted on graphs as follows:

