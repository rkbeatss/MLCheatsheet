"""

Minimizing MSE cost function [with Gradient Descent] approach to Linear Regression (multivariate and univariate)

"""
# Author: Rupsi Kaushik 

import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def cost_function(self, X, Y, slope, intercept):
        observations = len(X)
        total_error = 0.0
        for i in range(observations):
            total_error += (Y[i] - slope * X[i] + intercept) **2 
        return total_error / observations
    def update_weights(self, X, Y, slope, intercept, learning_rate):
        slope_deriv = 0
        intercept_deriv = 0
        observations = len(X)
        for i in range(observations):
            # partial derivatives for linear model slope = -2x(target - predicted)
            slope_deriv += -2*(X[i])*(Y[i] - (slope*X[i] + intercept))
            # partial derivative for bias -2(target - predicted)
            intercept_deriv += -2 * (Y[i] - slope * (X[i] + intercept))
        slope = (slope - (slope_deriv/observations)) * learning_rate
        intercept = (intercept - (intercept_deriv/ observations)) * learning_rate
        return slope, intercept



print(GradientDescent().cost_function(np.array([1,2,4]), np.array([3,5,6]), 3, 6))