# AssessLearner Report

## Does overfitting occur with respect to leaf_size? 

Consider the dataset istanbul.csv with DTLearner. Forwhich values of leaf_size does overfitting occur? Use RMSE as your metric for assessing overfitting.

To show whether overfitting occurs with respect to leaf size, two pieces of code were written.

1) The first piece of code was created to run the DTLearner algorithm over and over, 100 times against the /data/istanbul.csv data file and for each to record the RMSE and the collation for each.

Theoretically, it should be that overfitting occurs most in-sample when the leaf size is equal to one, because this should fit the data perfectly. Then, as the leaf size increases,we should see the leaf the error rate increasing.

These can then be compared with a similar plot with the out-sample data and we should see that the RMSE starts high, but then drops down to an optimal position before moving back up again.

The results of these were then plotted on graphs as follows:

![Plot 1](https://github.com/Liveo123/AssessLearner/blob/master/plot1.png)

As can be seen, the out-sample line does drop down rapidly at first and reaches optimal RMSE values of 0.0057 between leaf sizes of around 10 and 20 and then start to increase again.

A look at the correlation for the same run of code shows the corresponding results:

![Plot 2](https://github.com/Liveo123/AssessLearner/blob/master/plot2.png)

Here we can see the correlation starting as a perfect 1 for leaf-size 1, as would be expected for overfitting and then loosening up as the leaf size increases.

The in-sample correlation starts off at around 0.72, increases to is optimal value around 0.78 with a leaf size of between 5 and 10 and then drops back down.

When we take the same data and run cross validation on it (split the data up into different test and train groups and find the average of the results) we find that the out of sample RMSE drops a little as the leaf size increases and then goes back up again. The In-sample starts low and increases. They both stabilise around 0.006.

![Plot 3](https://github.com/Liveo123/AssessLearner/blob/master/plot3.png)

Examining the correlations for the same:

![Plot 4](https://github.com/Liveo123/AssessLearner/blob/master/plot4.png)



## Can bagging reduce or eliminate overfitting with respect to leaf_size?

Can bagging reduce or eliminate overfitting with respect to leaf_size? Again consider the dataset istanbul.csv with DTLearner. To investigate this choose a fixed number of bags to use and vary leaf_size to evaluate. Provide charts and/or tables to validate your conclusions.

To examine whether bagging reduces overfitting with respect to leaf size, a program has been created which runs the baglearner for the DTLearner. Here we can see that the in-sample overfitting (RMSE) goes down from 0.005 to around 0.010 over a range of 100 leaf size. 

![Plot 5](https://github.com/Liveo123/AssessLearner/blob/master/plot5.png)

When compared with the out of sample result, the overfitting changes very little over the same 100 leaf size difference.

This shows that bagging make little difference to overfitting when the leaf size is small, but increasing the leaf size shows that it makes a siginicant difference, changing little as it increases.

## Quantitatively compare "classic" decision trees (DTLearner) versus random trees (RTLearner). In which
ways is one method better than the other?

When the DTLearner is run against the RTLearner for RMSE on the Istanbul data, there seems to be very little difference, the RTLearner having a approximatley 0.002 more RMSE, which fades down to almost zero when the leaf size increase.

The correlation also gives a similar reading, with the RTLearner correlating slighlty less than the DTLearner.

So, the RTLearner did not get rid of overfitting when it is used in a situtation when a non-bag learner is used.

![Plot 6](https://github.com/Liveo123/AssessLearner/blob/master/plot6.png)


