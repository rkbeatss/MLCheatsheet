"""

Ordinary Least Squares coefficient estimates approach to simple Linear Regression

"""
# Author: Rupsi Kaushik 

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Inputs
    ----------
    X = [x1, x2, ... , xn] vector of predictors 
    Y = [y1, y2, ... , yn] vector of target values

    """
    def calculate_coefficients(self, X, Y):
        # calculate means 
        y_mean, x_mean = np.mean(Y), np.mean(X)
        # calculate slope (summation of (xi - x-mean) * (yi - y-mean)) / (summation of (xi - x-mean) squared)) 
        slope = sum((X - x_mean) * (Y - y_mean)) / sum((X- x_mean)**2)
        # calculate intercept which is intercept = y_mean - (slope * x_mean)
        intercept = y_mean - slope * x_mean
        return(intercept, slope)
    def plot_points(self, X, Y, coefficients):
        # create scatterplot of points 
        plt.scatter(X, Y, c = "black")
        # equation of regression line 
        y_pred = coefficients[0] + coefficients[1] * X
        # plot the line 
        plt.plot(X, y_pred, c = "red")
        plt.xlabel("Feature")
        plt.ylabel("Response")
        # show plot 
        plt.show()
    def predict(self, X, slope, intercept):
        return slope*X + intercept 