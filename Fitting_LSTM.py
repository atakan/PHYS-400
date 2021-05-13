#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Date: 2 April 2021
# Author: Çağatay Eskin
import matplotlib.pyplot as plt
import numpy as np

# FITTING A MODEL TO DATA USING GENERAL SOLUTION #
#------------------------------------------------#

N_train = 75 # number of data points
sigma_X = np.random.uniform(0.05, 0.1, N_train) # sample x coordinate sigmas from uniform distribution
sigma_Y = np.random.uniform(0.05, 0.1, N_train) # sample y coordinate sigmas from uniform distribution
noise_X = np.multiply(sigma_X, np.random.randn(N_train)) # error on x data (normally distributed)
noise_Y = np.multiply(sigma_Y, np.random.randn(N_train)) # error on y data (normally distributed)

# True parameters of the model which will be used to generate data
a_true = 1.9
b_true = 3.1

X = np.linspace(0.1, 0.9, N_train) + noise_X # observed X points
Y = b_true*(X-noise_X) + a_true + noise_Y # observed Y points (I think noise on X values must not effect Y, so
# I decided to create Y points without considering errors on X values (Gull-1989 pg. 1))

# Line with true parameters
x_plot = np.linspace(0.01, 1.1, N_train)
y_true_plot = b_true*x_plot + a_true


# Function that calculates the parameters of general solution (using same notation with York)
def linear_fit(b_pred, iter): # start with a prediction of b and number of iterations
    for k in range(iter):
        W = np.multiply(sigma_X, sigma_Y)/(b_pred**2*sigma_Y + sigma_X)
        X_bar = np.sum(np.multiply(W, X))/np.sum(W)
        Y_bar = np.sum(np.multiply(W, Y))/np.sum(W)
        U = X - X_bar
        V = Y - Y_bar
        alpha_numerator = 0
        alpha_denominator = 0
        beta_numerator = 0
        beta_denominator = 0
        gamma_numerator = 0
        gamma_denominator = 0
        for i in range(N_train):
            alpha_numerator = alpha_numerator + W[i]**2*U[i]*V[i]/sigma_X[i]
            alpha_denominator = alpha_denominator + W[i]**2*U[i]**2/sigma_X[i]
            beta_numerator = beta_numerator + (W[i]**2*V[i]**2/sigma_X[i]) - W[i]*U[i]**2
            beta_denominator = beta_denominator + W[i]**2*U[i]**2/sigma_X[i]
            gamma_numerator = gamma_numerator + W[i]*U[i]*V[i]
            gamma_denominator = gamma_denominator + W[i]**2*U[i]**2/sigma_X[i]
        alpha = (2*alpha_numerator)/(3*alpha_denominator)
        beta = beta_numerator/(3*beta_denominator)
        gamma = -gamma_numerator/gamma_denominator
        b = []
        for j in range(3): # there are three roots to be found
            b.append(alpha + 2*np.sqrt(alpha**2 - beta) *
                     np.cos(1/3*(np.arccos((alpha**3 - 3*alpha*beta/2 + gamma/2)/
                                           (alpha**2 - beta)**(3/2)) + 2*np.pi*j)))
        b_pred = b[2] # root we are looking for is generally the third one (if solution is unusual check that)

    a_pred = Y_bar - b_pred*X_bar # calculate the intercept
    return a_pred, b_pred

# Defining predicted parameters
a_fit, b_fit = linear_fit(5, 4) # input a reasonable b prediction (small values can cause error)
y_plot = a_fit + b_fit*x_plot


# FITTING A MODEL TO DATA WITH LSTM NETWORK#
#------------------------------------------#

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

# 1)Create Training Dataset

J = 3000 # number of training data

a_train = np.linspace(0.5, 3.5, J)
b_train = np.linspace(0.5, 3.5, J)

# Shuffling the arrays to get random a and b pairs
np.random.shuffle(a_train)
np.random.shuffle(b_train)

# Create empty arrays to fill for training data
training_features = np.zeros((J, 75, 4))
training_labels = np.zeros((J,2))

# Fill the tensors of training data
for j in range(J):
    sigma_train_x = np.random.uniform(0.05, 0.1, N_train) # sample x coordinate sigmas from uniform distribution
    sigma_train_y = np.random.uniform(0.05, 0.1, N_train) # sample y coordinate sigmas from uniform distribution
    noise_train_x = np.multiply(sigma_train_x, np.random.randn(N_train)) # error on x data (normally distributed)
    noise_train_y = np.multiply(sigma_train_y, np.random.randn(N_train)) # error on y data (normally distributed)
    x_train = np.linspace(0.1, 0.9, N_train) + noise_train_x
    y_train = a_train[j] + b_train[j] * (x_train-noise_train_x) + noise_train_y
    for i in range(N_train):
        for k in range(1):
            training_features[j,i,k] = x_train[i]
            training_features[j,i,k+1] = y_train[i]
            training_features[j,i,k+2] = sigma_train_x[i]
            training_features[j,i,k+3] = sigma_train_y[i]
    training_labels[j,0] = a_train[j]
    training_labels[j,1] = b_train[j]

# Prepare the data that we are going to predict parameters for
main = np.zeros((1, N_train, 4))
for i in range(N_train):
    main[0,i,0] = X[i]
    main[0,i,1] = Y[i]
    main[0,i,2] = sigma_X[i]
    main[0,i,3] = sigma_Y[i]


# 2)Define the neural net and its architecture then define the cost function and activation function
model = keras.Sequential([
    layers.InputLayer(input_shape = (None, 4)),
    layers.LSTM(300, activation = 'tanh'),
    layers.Dense(2),
])

# Define loss function and an optimizer
model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.001))

# Train the model
history = model.fit(
    training_features, training_labels,
    validation_split=0.2,
    verbose=1, epochs=50)

model.summary()

# Predict the model parameters
pred_DNN = model.predict(main)
a_fit_DNN = pred_DNN[0,0]
b_fit_DNN = pred_DNN[0,1]

# Create a figure
fig = plt.figure(figsize=(7, 6))
plt.style.use('seaborn-paper')
plt.rc('font', family='sans-serif')

# Plotting data (Choose either one of below)
plt.plot(X, Y, 'o', label='data', ms=8.5, alpha=0.3)
#plt.errorbar(X, Y, xerr=sigma_X, yerr=sigma_Y, fmt='o', label='data', ms=8.5, alpha=0.8) # error bars show SD

# Plotting the line with true parameters
plt.plot(x_plot, y_true_plot, '--', linewidth = 2.0, label=('line with true parameters (a=' + str(a_true) +
                                                            ', b=' + str(b_true) + ')'))

# Plotting linear fit
plt.plot(x_plot, y_plot, linewidth = 2.0, label=('general solution (a=' + str(np.round(a_fit, 3)) +
                                                 ', b=' + str(np.round(b_fit, 3)) + ')'))
# Plotting predictions of DNN
y_DNN = a_fit_DNN + b_fit_DNN*x_plot
plt.plot(x_plot, y_DNN, label = ('prediction of DNN (a=' + str(np.round(a_fit_DNN, 3)) + ', b=' + str(np.round(b_fit_DNN, 3)) + ')'), color = 'black')

# Plot specs
plt.tick_params(labelsize=11)
plt.xlabel('x', fontsize = 12)
plt.ylabel('y', fontsize = 12)
plt.title('Linear Fit to Data', fontsize = 12)
plt.legend(fontsize = 11)
plt.grid()
plt.show()
