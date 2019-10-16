The code and data in this directory were used for the experiments in the paper
''Online ARIMA Algorithms for Time Series Prediction''

1. Main scripts:
example.m: examples of how we solve problems

generate_data.m: code for generating our artificial data

arima_yk.m: the standard Yule-Wlaker estimation algorithm in an online setting

arima_ogd.m: ARIMA Online Gradient Descent algorithm

arima_ons.m: ARIMA Online Newton Step algorithm

2. Data sets
setting1.mat ~ setting4.mat are artificial data, setting5.mat and setting6.mat are real data. 

seq_dX is time series sequence under X-th order differencing.

3.Permission is granted for anyone to copy, use, modify, or distribute these programs and documents for any purpose.
As the programs were written for research purposes only, they have not been fully tested or validated. 
All use of these programs is entirely at the user's own risk.

