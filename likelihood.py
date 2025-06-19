import numpy as np
from tqdm import tqdm
import time

from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

def init_vertices_by_local_max(photon_counts, params, n_vertices, sigma_filter=[2, 2, 12]):
    smoothed = gaussian_filter(photon_counts.astype(np.float32), sigma=sigma_filter)
    
    # Find local maxima
    coordinates = peak_local_max(smoothed, 
                                 min_distance=1, 
                                 threshold_rel=0.001, 
                                 num_peaks=n_vertices,
                                 exclude_border=True)

    M = coordinates.shape[0]
    if M < n_vertices:
        # fallback: random for the rest
        extra = n_vertices - M
        extra_positions = np.random.rand(extra, 3) * [params.R, params.C, params.T]
        vertex_positions = np.vstack([coordinates, extra_positions])
    else:
        vertex_positions = coordinates[:n_vertices]
    
    vertex_positions = vertex_positions + 0.5

    return vertex_positions

def vertex_ll(vertex_positions, photon_counts, params, sigma, Xc, Yc, Zc, N):
    """
    Compute the log likelihood of the vertex positions given the photon counts.
    """
    
    expected_photons_per_vertex = params.E * params.P
    expected_counts = np.zeros((params.R, params.C, params.T))
    
    for i in range(N):
        vertex_pos = vertex_positions[i]
        
        x_term = ((Xc - vertex_pos[0])/sigma[0])**2
        y_term = ((Yc - vertex_pos[1])/sigma[1])**2
        z_term = ((Zc - vertex_pos[2])/sigma[2])**2
        
        probs = np.exp(-0.5 * (x_term + y_term + z_term)) / ((2*np.pi)**(3/2) * sigma[0] * sigma[1] * sigma[2])
        probs = probs / np.sum(probs)
        
        expected_counts += expected_photons_per_vertex * probs
    
    expected_counts = np.maximum(expected_counts, 1e-10)
    
    ll = np.sum(photon_counts * np.log(expected_counts) - expected_counts)
    
    return ll


def vertex_ll_gradient(vertex_positions, photon_counts, params, sigma, Xc, Yc, Zc, N):
    """
    Compute the gradient of the log likelihood of the vertex positions given the photon counts.
    """
    
    expected_photons_per_vertex = params.E * params.P
    expected_counts = np.zeros((params.R, params.C, params.T))
    
    # Store individual vertex contributions for gradient calculation
    vertex_contributions = []
    
    for i in range(N):
        vertex_pos = vertex_positions[i]
        
        x_term = ((Xc - vertex_pos[0])/sigma[0])**2
        y_term = ((Yc - vertex_pos[1])/sigma[1])**2
        z_term = ((Zc - vertex_pos[2])/sigma[2])**2
        
        probs = np.exp(-0.5 * (x_term + y_term + z_term)) / ((2*np.pi)**(3/2) * sigma[0] * sigma[1] * sigma[2])
        probs = probs / np.sum(probs)
        
        vertex_contribution = expected_photons_per_vertex * probs
        vertex_contributions.append(vertex_contribution)
        expected_counts += vertex_contribution
    
    expected_counts = np.maximum(expected_counts, 1e-10)
    ratio = photon_counts / expected_counts
    gradient = np.zeros_like(vertex_positions)
    
    for i in range(N):
        vertex_pos = vertex_positions[i]
        vertex_contribution = vertex_contributions[i]
        
        # Calculate gradient components for each dimension
        x_grad = vertex_contribution * (Xc - vertex_pos[0]) / (sigma[0]**2)
        y_grad = vertex_contribution * (Yc - vertex_pos[1]) / (sigma[1]**2)
        z_grad = vertex_contribution * (Zc - vertex_pos[2]) / (sigma[2]**2)
        
        gradient[i, 0] = np.sum(x_grad * ratio)
        gradient[i, 1] = np.sum(y_grad * ratio)
        gradient[i, 2] = np.sum(z_grad * ratio)
    
    return gradient


def compute_mle_vertex_positions(N, photon_counts, params, sigma, max_iterations=2000, tolerance=1e-1, learning_rate=0.01, verbose=False):
    """
    Compute the maximum likelihood estimates of the vertex positions given the photon counts.
    """
    
    # Set up grid coordinates
    x_grid = np.arange(params.R) + 0.5  # center of each bin
    y_grid = np.arange(params.C) + 0.5
    z_grid = np.arange(params.T) + 0.5
    Xc, Yc, Zc = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    vertex_positions = init_vertices_by_local_max(photon_counts, params, N)
    
    prev_ll = -np.inf
    max_ll = -np.inf
    if verbose:
        pbar = tqdm(range(max_iterations), desc="MLE optimization", ncols=100, leave=True)
    else:
        pbar = range(max_iterations)
    
    for iteration in pbar:
        gradient = vertex_ll_gradient(vertex_positions, photon_counts, params, sigma, Xc, Yc, Zc, N)
        gradient = np.clip(gradient, -20, 20)
        
        vertex_positions += learning_rate * gradient
        
        # Enforce boundary constraints
        vertex_positions = np.clip(vertex_positions, 
                                  [0.1, 0.1, 0.1], 
                                  [params.R - 0.1, params.C - 0.1, params.T - 0.1])
        
        expected_photons_per_vertex = params.E * params.P
        expected_counts = np.zeros((params.R, params.C, params.T))
        
        # Store individual vertex contributions for gradient calculation
        vertex_contributions = []
        
        for i in range(N):
            vertex_pos = vertex_positions[i]
            
            x_term = ((Xc - vertex_pos[0])/sigma[0])**2
            y_term = ((Yc - vertex_pos[1])/sigma[1])**2
            z_term = ((Zc - vertex_pos[2])/sigma[2])**2
            
            probs = np.exp(-0.5 * (x_term + y_term + z_term)) / ((2*np.pi)**(3/2) * sigma[0] * sigma[1] * sigma[2])
            probs = probs / np.sum(probs)
            
            vertex_contribution = expected_photons_per_vertex * probs
            vertex_contributions.append(vertex_contribution)
            expected_counts += vertex_contribution
        
        expected_counts = np.maximum(expected_counts, 1e-10)
        
        ll = np.sum(photon_counts * np.log(expected_counts) - expected_counts)
        
        ll_change = np.min([np.abs(ll - prev_ll), np.abs(ll - max_ll)])
        if verbose:
            pbar.set_postfix({"LL": f"{ll:.6f}", "Change": f"{ll_change:.6f}"})
        
        if ll_change < tolerance:
            if verbose:
                pbar.set_description(f"Converged after {iteration+1} iterations (LL = {ll:.6f})")
            break
            
        prev_ll = ll
        max_ll = np.max([max_ll, ll])
    
    # Calculate uncertainties as standard error of the mean
    uncertainties = np.zeros_like(vertex_positions)
    
    for i in range(N):
        vertex_pos = vertex_positions[i]
        
        x_term = ((Xc - vertex_pos[0])/sigma[0])**2
        y_term = ((Yc - vertex_pos[1])/sigma[1])**2
        z_term = ((Zc - vertex_pos[2])/sigma[2])**2
        
        weights = np.exp(-0.5 * (x_term + y_term + z_term))
        weights = weights * photon_counts
        N_eff = np.sum(weights)
        
        if N_eff > 0:
            # Standard error = sigma / sqrt(N)
            uncertainties[i, 0] = sigma[0] / np.sqrt(N_eff)
            uncertainties[i, 1] = sigma[1] / np.sqrt(N_eff)
            uncertainties[i, 2] = sigma[2] / np.sqrt(N_eff)
        else:
            uncertainties[i] = [np.nan, np.nan, np.nan]
    
    return vertex_positions, uncertainties, ll