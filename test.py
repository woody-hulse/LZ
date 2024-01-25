import numpy as np 
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt

mean = 10
  
# Let's create a function to model and create data 
def func(x, a, sigma): 
    return a*np.exp(-(x-mean)**2/(2*sigma**2)) 
  
# Generating clean data 
x = np.linspace(0, 20, 1000) 
y = func(x, 1, 2) 
  
# Adding noise to the data 
yn = y + 0.2 * np.random.normal(size=len(x)) 
  
# Plot out the current state of the data and model 
plt.plot(x, y, c='k', label='Function') 
plt.scatter(x, yn) 
  
# Executing curve_fit on noisy data 
popt, pcov = curve_fit(func, x, yn) 
  
#popt returns the best fit values for parameters of the given model (func) 
print (popt) 
  
ym = func(x, popt[0], popt[1]) 
plt.plot(x, ym, c='r', label='Best fit') 
plt.legend() 
plt.show()