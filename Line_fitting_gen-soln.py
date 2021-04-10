# Date: 2 April 2021
# Author: Çağatay Eskin

import matplotlib.pyplot as plt
import numpy as np

N_train = 15 # number of training data
sigma_X = np.random.uniform(0.05, 0.15, N_train) # sample x coordinate errors from uniform distribution
sigma_Y = np.random.uniform(0.05, 0.15, N_train) # sample y coordinate errors from uniform distribution
noise_X = np.multiply(sigma_X, np.random.randn(N_train)) 
noise_Y = np.multiply(sigma_Y, np.random.randn(N_train))

a_true = -2.7
b_true = 3.4

X = np.linspace(0.1, 0.99, N_train) + noise_X # observed X points
Y = b_true*X + a_true + noise_Y # observed Y points

fig = plt.figure(figsize=(7, 6))
plt.style.use('seaborn-paper')
plt.rc('font', family='sans-serif')
# colors: 'darkmagenta','indigo','mediumslateblue','steelblue','teal','turquoise','orchid'

# Plotting data
plt.errorbar(X, Y, xerr=sigma_X, yerr=sigma_Y, fmt='o', label='data', ms=8.5, alpha=0.8) # error bars show SD

# Plotting the line with true parameters
x_plot = np.linspace(0.01, 1.1, 100)
y_true_plot = b_true*x_plot + a_true
plt.plot(x_plot, y_true_plot, '--', linewidth = 2.0, label=('line with true parameters (a=' + str(a_true) +
                                                            ', b=' + str(b_true) + ')'))

iter = 4 # number of iterations

# Function that calculates the parameters (using same notation with York)
def linear_fit(b_pred): # start with a prediction of b (b_pred)
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
        b_pred = b[2] # root we are looking for is generally the third one
        
    a_pred = Y_bar - b_pred*X_bar # calculate the intercept
    return a_pred, b_pred

# Defining predicted parameters
a_fit, b_fit = linear_fit(5) # with and initial value of b_pred

# Plotting linear fit
y_plot = a_fit + b_fit*x_plot
plt.plot(x_plot, y_plot, linewidth = 2.0, label=('linear fit (a=' + str(np.round(a_fit, 3)) + 
                                                 ', b=' + str(np.round(b_fit, 3)) + ')'))
# Plot specs
plt.tick_params(labelsize=11)
plt.xlabel('x', fontsize = 12)
plt.ylabel('y', fontsize = 12)
plt.title('Linear Fit to Data', fontsize = 12)
plt.legend(fontsize = 11)
plt.grid()
plt.show()
