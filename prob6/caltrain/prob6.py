"""
First: 
Load the dataset into Python.
Fit a Linear Regression model on the training data.
Make predictions on the test data.
Report the root-mean-squared error, or RMSE, of your model on the test data for the answer checker to verify.

What we want to predict: the expected number of people who will ride the caltrain during a one-hour period, 
given the information in the features above. The recorded number of passengers for each datapoint is stored 
in the dataset under the column labeled passengers_per_hour.



"""

import csv
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression


def main():
    train_data = csv.reader(open('caltrain-train.csv'))
    #create a 2D array of inputs, where we have 7 features and 1 label, which is the passengers_per_hour
    next(train_data)
    train_rows = list(train_data)
    num_rows = len(train_rows)

    X = np.zeros((num_rows, 7))
    y = np.zeros(num_rows)
    for i, row in enumerate(train_rows):
        X[i, 0] = float(row[0])
        X[i, 1] = float(row[1])
        X[i, 2] = float(row[2])
        X[i, 3] = float(row[3])
        X[i, 4] = float(row[4])
        X[i, 5] = float(row[5])
        X[i, 6] = float(row[6])
        y[i] = float(row[7])
    # the label is the passengers_per_hour


    test_data = csv.reader(open('caltrain-test.csv'))
    # Skip header row
    next(test_data)
    test_rows = list(test_data)
    num_rows_new = len(test_rows)
    new_X = np.zeros((num_rows_new, 7))
    test_y = np.zeros(num_rows_new)
    for i, row in enumerate(test_rows):
        new_X[i, 0] = float(row[0])
        new_X[i, 1] = float(row[1])
        new_X[i, 2] = float(row[2])
        new_X[i, 3] = float(row[3])
        new_X[i, 4] = float(row[4])
        new_X[i, 5] = float(row[5])
        new_X[i, 6] = float(row[6])
        test_y[i] = float(row[7])

    # Create the model!
    model = LinearRegression()

    # Train the model
    # (y = the labels, X = a 2D array of inputs)
    model.fit(X, y)
    
    pred_y = model.predict(new_X)
    error = pred_y - test_y
    # compute the RMSE on test data
    rmse = np.sqrt(np.mean((error) ** 2))
    print(f"RMSE: {rmse}")

    # representing the difference between a model's prediction and the true value
    sigma_squared = np.mean(error ** 2)
    print(f"MLE estimate: {sigma_squared}")

    # the distribution of error is the normal distribution with mean 0 and variance (sigma_squared)
    error_distribution = stats.norm(0,np.sqrt(sigma_squared))

    #prob that error is greater than 20, which is 1 - P(error <= 20)
    probability = 1-error_distribution.cdf(20)
    print(f"Probability: {probability*2}")


if __name__ == "__main__":
    main()