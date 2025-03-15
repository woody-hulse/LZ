import numpy as np
import multiprocessing
import time

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
DIFFWIDTH       = 300   # electron arrival time (ns)
G2              = 47.35 # photon count poisson distribution
EGASWIDTH       = 450   # photon arrival time (ns)
PHDWIDTH        = 10    # photon PMT activation (ns)
SAMPLERATE      = 10    # PMT sample interval (ns)
MAX_ELECTRON    = 200   # extreme of electrons for maximum energy deposition
DENSITY         = 0.5   # density parameter for neutron impact

NEUTRON_WEIGHT  = 4.0   # neutron weight
XE_WEIGHT       = 131.0 # xenon weight

class PMTArray:
    def __init__(self, array, bin_function, boundary_function, photon_function):
        self.array = array
        self.bin_function = bin_function
        self.boundary_function = boundary_function
        self.photon_function = photon_function

def gaussian(x, C, mu, sigma):
    return C*np.exp(-(x-mu)**2/(2*sigma**2))

def simulate_event(array):
    
    def simulate_neutron(x, y, z, theta, phi):
        dist = np.random.exponential(DENSITY)

        x = x + dist*np.sin(theta)*np.cos(phi)
        y = y + dist*np.sin(theta)*np.sin(phi)
        z = z + dist*np.cos(theta)

        theta = np.arccos(1-2*np.random.rand())
        phi = 2*np.pi*np.random.rand()

        return x, y, z, theta, phi

    def compute_energy_deposition(theta1, theta2, phi1, phi2):
        vx1 = np.sin(theta1)*np.cos(phi1)
        vy1 = np.sin(theta1)*np.sin(phi1)
        vz1 = np.cos(theta1)

        vx2 = np.sin(theta2)*np.cos(phi2)
        vy2 = np.sin(theta2)*np.sin(phi2)
        vz2 = np.cos(theta2)

        reflected_angle = np.arccos(vx1*vx2 + vy1*vy2 + vz1*vz2)

        return (1 - np.cos(reflected_angle)) / 2 * NEUTRON_WEIGHT / XE_WEIGHT 

    x, y, z, r, theta, phi = array.photon_function()

    for i in range(10):
        x, y, z, theta_, phi_ = simulate_neutron(x, y, z, theta, phi)
        energy_deposition = compute_energy_deposition(theta, theta_, phi, phi_)
        theta, phi = theta_, phi_
        r = (1 - energy_deposition) * r
        
        electron_count_mean = MAX_ELECTRON * energy_deposition * XE_WEIGHT / NEUTRON_WEIGHT
        electron_count = np.random.poisson(electron_count_mean)

        


    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        x, y, z, theta_, phi_ = simulate_neutron(x, y, z, theta, phi)
        energy_deposition = compute_energy_deposition(theta, theta_, phi, phi_)
        theta, phi = theta_, phi_
        # if not array.boundary_function(x, y, z):
        #     break

        vx = np.sin(theta)*np.cos(phi) / 10
        vy = np.sin(theta)*np.sin(phi) / 10
        vz = np.cos(theta) / 10

        ax.scatter(x, y, z, c='blue', marker='o', s=10)
        ax.quiver(x, y, z, vx, vy, vz, color='blue', arrow_length_ratio=0.2)
        ax.text(x, y, z, f'{i}', color='red', fontsize=12)

        print(f'{i + 1}: ', (x, y, z), (theta, phi))
        print(f'energy deposition: {energy_deposition}')
    
    plt.show()


def square_bin_function(x, y, z):
    return np.floor(x), np.floor(y), np.floor(z)

def square_boundary_function(x, y, z, width=24, height=24):
    if isinstance(x, np.ndarray):
        return np.logical_and(0 < x, x < width) and np.logical_and(0 < y, y < width) and np.logical_and(0 < z, z < height)
    return 0 < x < width and 0 < y < width and 0 < z < height

def square_photon_function(count=1, width=24, height=24):
    x = np.zeros(count) if count > 1 else 0
    y = np.random.normal(width / 2, np.sqrt(width / 2), count) if count > 1 else np.random.normal(width / 2, np.sqrt(width / 2))
    z = np.random.normal(height / 2, np.sqrt(height / 2), count) if count > 1 else np.random.normal(height / 2, np.sqrt(height / 2))
    theta, phi = 0, 0
    r = 1
    return x, y, z, r, theta, phi

def main():
    array = PMTArray(
        array               = np.zeros((24, 24, 700)), 
        bin_function        = square_bin_function, 
        boundary_function   = square_boundary_function, 
        photon_function     = square_photon_function
    )

    simulate_event(array)



if __name__ == '__main__':

    main()
    '''
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(1000):
        theta = np.arccos(1-2*np.random.rand())
        phi = 2*np.pi*np.random.rand()

        # Plot as 3d scatterpoint with radius 1
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)

        ax.scatter(x, y, z, c='blue', marker='o', s=10)
    
    plt.show()
    '''