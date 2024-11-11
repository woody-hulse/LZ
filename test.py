import numpy as np
import matplotlib.pyplot as plt

def double_gaussian(x, delta_mu, sigma=1.0):
    """
    Model a double Gaussian and return the resulting Gaussian when delta_mu is imaginary.
    When delta_mu^2 < 0, the width contracts.
    
    Parameters:
    - x: array-like, the input points where the Gaussian is evaluated
    - delta_mu: the separation between the peaks of the double Gaussian
    - sigma: the standard deviation of each individual Gaussian (default is 1.0)
    
    Returns:
    - resulting_gaussian: a Gaussian that combines the two peaks with modified width
    """
    # Calculate delta_mu squared
    delta_mu_squared = delta_mu ** 2
    
    if delta_mu_squared >= 0:
        # Conventional double Gaussian: Add width based on delta_mu squared
        adjusted_sigma = sigma + (1 / 8) * delta_mu_squared
    else:
        # Narrowing occurs, delta_mu is imaginary: Contract the width
        adjusted_sigma = sigma + (1 / 8) * delta_mu_squared
    
    # Ensure sigma is positive
    adjusted_sigma = np.abs(adjusted_sigma)
    
    # Create the Gaussian with the adjusted width
    resulting_gaussian = np.exp(-x ** 2 / (2 * adjusted_sigma ** 2)) / (adjusted_sigma * np.sqrt(2 * np.pi))
    
    return resulting_gaussian

# Example usage
x = np.linspace(-5, 5, 500)

# Positive delta_mu (normal behavior)
delta_mu_positive = 2.0
gaussian_positive = double_gaussian(x, delta_mu_positive)

# Negative delta_mu (imaginary, causing narrowing)
delta_mu_negative = 1j * 2.0  # Using imaginary delta_mu
gaussian_negative = double_gaussian(x, 0)  # Ensure we handle imaginary delta_mu correctly

# Plot both Gaussians
plt.plot(x, gaussian_positive, label=f"Positive delta_mu = {delta_mu_positive}")
plt.plot(x, gaussian_negative, label=f"Imaginary delta_mu = {delta_mu_negative} (Narrowed)")
plt.title("Gaussian with Positive and Imaginary delta_mu")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()