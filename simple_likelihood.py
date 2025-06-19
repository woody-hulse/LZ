import numpy as np
from math import log, pi, sqrt
from scipy.stats import poisson, multivariate_normal
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from tqdm import tqdm

from simple_pulse import Params
from simple_pulse import simulate_vertex_electron_photon, simulate_vertex_electron_photon_explicit

from likelihood import compute_mle_vertex_positions

def plot_hit_comparison(hit1, hit2, filename='hit_comparison', title1='hit1', title2='hit2', colorbar1=False, colorbar2=False):
    print(f"Generating GIF: {filename}.gif")

    hit1 = np.transpose(hit1, axes=[2, 0, 1])
    hit2 = np.transpose(hit2, axes=[2, 0, 1])

    image_frames = []
    
    for t1, t2 in tqdm(zip(hit1, hit2), total=hit1.shape[0], desc="Creating frames", ncols=120):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        im1 = plt.imshow(t1, vmin=0, vmax=1, origin='upper', aspect='auto', cmap='hot')
        plt.title(title1)
        plt.axis('off')
        if colorbar1:
            plt.colorbar(im1, pad=0.01, fraction=0.046, shrink=0.8, aspect=20)
            
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(t2, vmin=0, vmax=1, origin='upper', aspect='auto', cmap='hot')
        plt.title(title2)
        plt.axis('off')
        if colorbar2:
            plt.colorbar(im2, pad=0.01, fraction=0.046, shrink=0.8, aspect=20)
        
        plt.tight_layout()
        
        plt.gcf().canvas.draw()
        width, height = plt.gcf().get_size_inches() * plt.gcf().dpi
        
        data = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        image_array = data.reshape(int(height), int(width), 3)
        
        image_frame = Image.fromarray(image_array)
        image_frames.append(image_frame)
        
        plt.close()
    
    gif_filename = f"{filename}.gif"
    image_frames[0].save(
        gif_filename,
        save_all=True,
        duration=20,
        loop=0,
        append_images=image_frames[1:]
    )
    print(f"GIF saved: {gif_filename}")
    

def plot_vertex_positions_comparison(true_positions, mle_positions, params, filename='vertex_positions_comparison'):
    """
    Creates a 3D scatterplot of the true vertex positions vs. the MLE vertex positions.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        true_positions[:, 0], true_positions[:, 1], true_positions[:, 2],
        marker='o', label='True Vertex Positions'
    )

    ax.scatter(
        mle_positions[:, 0], mle_positions[:, 1], mle_positions[:, 2],
        marker='^', label='MLE Vertex Positions'
    )
    
    ax.set_xlim([0, params.R])
    ax.set_ylim([0, params.C])
    ax.set_zlim([0, params.T])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('True vs. MLE Vertex Positions')
    ax.legend()

    full_filename = f"{filename}.png"

    plt.savefig(full_filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print(f"Scatterplot saved to: {full_filename}")


def compute_mle_vertex_positions_fim(photon_counts, params, sigma, max_iterations=100, tolerance=1e-4, initial_positions=None):
    """
    Compute the maximum likelihood estimate (MLE) of vertex positions given observed photon counts,
    with uncertainties derived from the Fisher Information Matrix.
    
    Parameters:
    -----------
    photon_counts : ndarray
        3D array of observed photon counts
    params : object
        Object containing parameters (N, R, C, T, E, P, rv)
    sigma : array-like
        Standard deviations of the Gaussian PSF in x, y, z dimensions
    max_iterations : int
        Maximum number of EM iterations
    tolerance : float
        Convergence tolerance for log-likelihood
    initial_positions : ndarray or None
        Initial vertex positions, if None random positions will be used
        
    Returns:
    --------
    vertex_positions : ndarray
        Estimated vertex positions
    uncertainties : ndarray
        Estimated uncertainties (standard deviations) for each vertex position
    log_likelihood : float
        Final log-likelihood value
    """
    
    x_grid = np.arange(params.R) + 0.5
    y_grid = np.arange(params.C) + 0.5
    z_grid = np.arange(params.T) + 0.5
    Xc, Yc, Zc = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    if initial_positions is None:
        vertex_positions = np.zeros((params.N, 3))
        for i in range(params.N):
            lower, upper = params.rv / 2, 1 - params.rv / 2
            vertex_positions[i, 0] = np.random.uniform(params.R * lower[0], params.R * upper[0])
            vertex_positions[i, 1] = np.random.uniform(params.C * lower[1], params.C * upper[1])
            vertex_positions[i, 2] = np.random.uniform(params.T * lower[2], params.T * upper[2])
    else:
        vertex_positions = initial_positions.copy()
    
    expected_photons_per_vertex = params.E * params.P
    
    prev_log_likelihood = -np.inf
    for iteration in tqdm(range(max_iterations), desc="FIM MLE iterations", ncols=120):
        responsibilities = np.zeros((params.N, params.R, params.C, params.T))
        expected_counts = np.zeros((params.R, params.C, params.T))
        
        for i in range(params.N):
            vertex_pos = vertex_positions[i]
            
            x_term = ((Xc - vertex_pos[0])/sigma[0])**2
            y_term = ((Yc - vertex_pos[1])/sigma[1])**2
            z_term = ((Zc - vertex_pos[2])/sigma[2])**2
            
            probs = np.exp(-0.5 * (x_term + y_term + z_term)) / ((2*np.pi)**(3/2) * sigma[0] * sigma[1] * sigma[2])
            probs = probs / np.sum(probs)
            
            vertex_expected = expected_photons_per_vertex * probs
            expected_counts += vertex_expected
            
            responsibilities[i] = vertex_expected
        
        # Avoid division by zero
        expected_counts = np.maximum(expected_counts, 1e-10)
        
        # Calculate responsibilities
        for i in range(params.N):
            responsibilities[i] = responsibilities[i] / expected_counts * photon_counts
        
        # M-step: Update vertex positions and calculate uncertainties
        new_positions = np.zeros_like(vertex_positions)
        uncertainties = np.zeros_like(vertex_positions)
        
        for i in range(params.N):
            weights = responsibilities[i]
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                # Update position (weighted mean)
                new_positions[i, 0] = np.sum(weights * Xc) / total_weight
                new_positions[i, 1] = np.sum(weights * Yc) / total_weight
                new_positions[i, 2] = np.sum(weights * Zc) / total_weight
                
                # Calculate effective number of photons for this vertex
                # This represents how many photons are effectively contributing to this vertex's position estimate
                N_eff = np.sum(responsibilities[i])
                
                # Calculate Fisher Information Matrix diagonal elements
                # For a Gaussian PSF, FIM_ii = N_eff / sigma_i^2
                FIM_xx = N_eff / (sigma[0]**2)
                FIM_yy = N_eff / (sigma[1]**2)
                FIM_zz = N_eff / (sigma[2]**2)
                
                # Cramér-Rao lower bound: standard deviation = sqrt(1/FIM_ii)
                uncertainties[i, 0] = np.sqrt(1.0 / max(FIM_xx, 1e-10))
                uncertainties[i, 1] = np.sqrt(1.0 / max(FIM_yy, 1e-10))
                uncertainties[i, 2] = np.sqrt(1.0 / max(FIM_zz, 1e-10))
            else:
                # If no photons are assigned to this vertex, keep the position and set uncertainty to NaN
                new_positions[i] = vertex_positions[i]
                uncertainties[i] = [np.nan, np.nan, np.nan]
        
        # Calculate log-likelihood
        log_likelihood = np.sum(photon_counts * np.log(expected_counts) - expected_counts)
        
        # Check for convergence
        if np.abs(log_likelihood - prev_log_likelihood) < tolerance:
            break
        
    prev_log_likelihood = log_likelihood
    vertex_positions = new_positions
        
    return vertex_positions, uncertainties, log_likelihood

'''
def compute_mle_vertex_positions(photon_counts, params, sigma, max_iterations=100, tolerance=1e-4, initial_positions=None):
    """
    Compute the maximum likelihood estimate (MLE) of vertex positions given observed photon counts.
    """
    
    x_grid = np.arange(params.R) + 0.5 # center of each bin
    y_grid = np.arange(params.C) + 0.5
    z_grid = np.arange(params.T) + 0.5
    Xc, Yc, Zc = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    if initial_positions is None:
        vertex_positions = np.zeros((params.N, 3))

        smoothed_counts = gaussian_filter(gaussian_filter(photon_counts.astype(float), sigma=5), sigma=5)
        neighborhood = generate_binary_structure(3, 1)
        local_max = (smoothed_counts == maximum_filter(smoothed_counts, footprint=neighborhood, mode='constant', cval=np.inf))
        threshold = np.mean(smoothed_counts) * 0.1  # adjust this factor as needed
        local_max = local_max & (smoothed_counts > threshold)

        # Plot the local maxima for visualization
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111, projection='3d')
        ax.scatter(Xc[local_max], Yc[local_max], Zc[local_max], c='red', marker='o', s=50, label='Local Maxima')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Local Maxima in Smoothed Photon Counts')

        plt.show()
        
        maxima_coords = np.array(np.where(local_max)).T
        maxima_values = smoothed_counts[local_max]
        
        sorted_indices = np.argsort(-maxima_values)
        num_maxima = min(params.N, len(maxima_values))
        
        for i in range(num_maxima):
            idx = sorted_indices[i]
            vertex_positions[i, 0] = maxima_coords[idx, 0] + 0.5
            vertex_positions[i, 1] = maxima_coords[idx, 1] + 0.5
            vertex_positions[i, 2] = maxima_coords[idx, 2] + 0.5
            
        if num_maxima < params.N:
            print('not enough maxima, filling with random positions')
            for i in range(num_maxima, params.N):
                lower, upper = params.rv / 2, 1 - params.rv / 2
                vertex_positions[i, 0] = np.random.uniform(params.R * lower[0], params.R * upper[0])
                vertex_positions[i, 1] = np.random.uniform(params.C * lower[1], params.C * upper[1])
                vertex_positions[i, 2] = np.random.uniform(params.T * lower[2], params.T * upper[2])
    
        print(vertex_positions)
    else:
        vertex_positions = initial_positions.copy()
    
    expected_photons_per_vertex = params.E * params.P
    
    prev_log_likelihood = -np.inf
    pbar = tqdm(range(max_iterations), desc="MLE optimization", ncols=100, leave=True)
    for iteration in pbar:
        responsibilities = np.zeros((params.N, params.R, params.C, params.T))
        
        expected_counts = np.zeros((params.R, params.C, params.T))
        for i in range(params.N):
            vertex_pos = vertex_positions[i]
            
            # Use anisotropic Gaussian with separate sigmas for x, y, z
            x_term = ((Xc - vertex_pos[0])/sigma[0])**2
            y_term = ((Yc - vertex_pos[1])/sigma[1])**2
            z_term = ((Zc - vertex_pos[2])/sigma[2])**2
            
            probs = np.exp(-0.5 * (x_term + y_term + z_term)) / ((2*np.pi)**(3/2) * sigma[0] * sigma[1] * sigma[2])
            
            probs = probs / np.sum(probs)
            
            vertex_expected = expected_photons_per_vertex * probs
            expected_counts += vertex_expected
            
            responsibilities[i] = vertex_expected
        
        expected_counts = np.maximum(expected_counts, 1e-10)
        
        # for i in range(params.N):
        #     responsibilities[i] = responsibilities[i] / expected_counts * photon_counts
        
        new_positions = np.zeros_like(vertex_positions)
        uncertainties = np.zeros_like(vertex_positions)
        
        for i in range(params.N):
            weights = responsibilities[i]
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                new_positions[i, 0] = np.sum(weights * Xc) / total_weight
                new_positions[i, 1] = np.sum(weights * Yc) / total_weight
                new_positions[i, 2] = np.sum(weights * Zc) / total_weight
                
                uncertainties[i, 0] = np.sqrt(np.sum(weights * (Xc - new_positions[i, 0])**2) / total_weight)
                uncertainties[i, 1] = np.sqrt(np.sum(weights * (Yc - new_positions[i, 1])**2) / total_weight)
                uncertainties[i, 2] = np.sqrt(np.sum(weights * (Zc - new_positions[i, 2])**2) / total_weight)
            else:
                new_positions[i] = vertex_positions[i]
                uncertainties[i] = [np.nan, np.nan, np.nan]
        
        log_likelihood = np.sum(photon_counts * np.log(expected_counts) - expected_counts)
        ll_change = np.abs(log_likelihood - prev_log_likelihood)
        
        pbar.set_postfix({"LL": f"{log_likelihood : .6f}"})
        
        if ll_change < tolerance:
            pbar.set_description(f"Converged after {iteration+1} iterations (LL = {log_likelihood : .6f})")
            break
            
        prev_log_likelihood = log_likelihood
        vertex_positions = new_positions
    
    return vertex_positions, uncertainties, log_likelihood
'''


def compute_mle_vertex_positions_covariance(photon_counts, params, sigma, max_iterations=100, tolerance=1e-4, initial_positions=None):
    """
    Compute the maximum likelihood estimate (MLE) of vertex positions given observed photon counts,
    with uncertainties derived from the empirical covariance of assigned photons.
    
    Parameters:
    -----------
    photon_counts : ndarray
        3D array of observed photon counts
    params : object
        Object containing parameters (N, R, C, T, E, P, rv)
    sigma : array-like
        Standard deviations of the Gaussian PSF in x, y, z dimensions
    max_iterations : int
        Maximum number of EM iterations
    tolerance : float
        Convergence tolerance for log-likelihood
    initial_positions : ndarray or None
        Initial vertex positions, if None random positions will be used
        
    Returns:
    --------
    vertex_positions : ndarray
        Estimated vertex positions
    uncertainties : ndarray
        Estimated uncertainties (standard deviations) for each vertex position
    log_likelihood : float
        Final log-likelihood value
    """
    
    # Set up grid coordinates
    x_grid = np.arange(params.R) + 0.5  # center of each bin
    y_grid = np.arange(params.C) + 0.5
    z_grid = np.arange(params.T) + 0.5
    Xc, Yc, Zc = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    # Initialize vertex positions
    if initial_positions is None:
        vertex_positions = np.zeros((params.N, 3))
        for i in range(params.N):
            lower, upper = params.rv / 2, 1 - params.rv / 2
            vertex_positions[i, 0] = np.random.uniform(params.R * lower[0], params.R * upper[0])
            vertex_positions[i, 1] = np.random.uniform(params.C * lower[1], params.C * upper[1])
            vertex_positions[i, 2] = np.random.uniform(params.T * lower[2], params.T * upper[2])
    else:
        vertex_positions = initial_positions.copy()
    
    expected_photons_per_vertex = params.E * params.P
    
    # EM algorithm
    prev_log_likelihood = -np.inf
    for iteration in range(max_iterations):
        # E-step: Calculate responsibilities and expected counts
        responsibilities = np.zeros((params.N, params.R, params.C, params.T))
        expected_counts = np.zeros((params.R, params.C, params.T))
        
        for i in range(params.N):
            vertex_pos = vertex_positions[i]
            
            # Calculate Gaussian PSF probabilities
            x_term = ((Xc - vertex_pos[0])/sigma[0])**2
            y_term = ((Yc - vertex_pos[1])/sigma[1])**2
            z_term = ((Zc - vertex_pos[2])/sigma[2])**2
            
            probs = np.exp(-0.5 * (x_term + y_term + z_term)) / ((2*np.pi)**(3/2) * sigma[0] * sigma[1] * sigma[2])
            probs = probs / np.sum(probs)
            
            vertex_expected = expected_photons_per_vertex * probs
            expected_counts += vertex_expected
            
            responsibilities[i] = vertex_expected
        
        # Avoid division by zero
        expected_counts = np.maximum(expected_counts, 1e-10)
        
        # Calculate responsibilities
        for i in range(params.N):
            responsibilities[i] = responsibilities[i] / expected_counts * photon_counts
        
        # M-step: Update vertex positions and calculate uncertainties
        new_positions = np.zeros_like(vertex_positions)
        uncertainties = np.zeros_like(vertex_positions)
        covariance_matrices = np.zeros((params.N, 3, 3))
        
        for i in range(params.N):
            weights = responsibilities[i]
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                # Update position (weighted mean)
                new_positions[i, 0] = np.sum(weights * Xc) / total_weight
                new_positions[i, 1] = np.sum(weights * Yc) / total_weight
                new_positions[i, 2] = np.sum(weights * Zc) / total_weight
                
                # Calculate empirical covariance matrix
                # First, create arrays of deviations from the mean
                dx = Xc - new_positions[i, 0]
                dy = Yc - new_positions[i, 1]
                dz = Zc - new_positions[i, 2]
                
                # Calculate weighted covariance matrix elements
                cov_xx = np.sum(weights * dx * dx) / total_weight
                cov_xy = np.sum(weights * dx * dy) / total_weight
                cov_xz = np.sum(weights * dx * dz) / total_weight
                cov_yy = np.sum(weights * dy * dy) / total_weight
                cov_yz = np.sum(weights * dy * dz) / total_weight
                cov_zz = np.sum(weights * dz * dz) / total_weight
                
                # Construct the covariance matrix
                cov_matrix = np.array([
                    [cov_xx, cov_xy, cov_xz],
                    [cov_xy, cov_yy, cov_yz],
                    [cov_xz, cov_yz, cov_zz]
                ])
                
                # Store the covariance matrix
                covariance_matrices[i] = cov_matrix
                
                # Standard errors are the square roots of the diagonal elements
                # Adjusted by sqrt(N) to account for effective sample size
                n_eff = total_weight  # Effective number of photons
                
                # Standard error of the mean = std / sqrt(n)
                uncertainties[i, 0] = np.sqrt(cov_xx / n_eff)
                uncertainties[i, 1] = np.sqrt(cov_yy / n_eff)
                uncertainties[i, 2] = np.sqrt(cov_zz / n_eff)
            else:
                # If no photons are assigned to this vertex, keep the position and set uncertainty to NaN
                new_positions[i] = vertex_positions[i]
                uncertainties[i] = [np.nan, np.nan, np.nan]
                covariance_matrices[i] = np.eye(3) * np.nan
        
        # Calculate log-likelihood
        log_likelihood = np.sum(photon_counts * np.log(expected_counts) - expected_counts)
        
        # Check for convergence
        if np.abs(log_likelihood - prev_log_likelihood) < tolerance:
            break
            
        prev_log_likelihood = log_likelihood
        vertex_positions = new_positions
    
    return vertex_positions, uncertainties, log_likelihood #, covariance_matrices


def model_expected_counts(x_grid, y_grid, z_grid, vertex_positions, params, sigma, 
                          bin_width=1.0, bin_height=1.0, bin_depth=1.0):
    """
    Model the expected photon counts in a 3D grid given vertex positions.
    """
    expected_counts = np.zeros((len(x_grid), len(y_grid), len(z_grid)))

    Xc, Yc, Zc = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    expected_photons_per_vertex = params.E * params.P
    
    for vertex_pos in tqdm(vertex_positions, desc="Modeling expected counts", ncols=120):
        # Use anisotropic Gaussian with separate sigmas for x, y, z
        x_term = ((Xc - vertex_pos[0])/sigma[0])**2
        y_term = ((Yc - vertex_pos[1])/sigma[1])**2
        z_term = ((Zc - vertex_pos[2])/sigma[2])**2
        
        probs = np.exp(-0.5 * (x_term + y_term + z_term)) / ((2*np.pi)**(3/2) * sigma[0] * sigma[1] * sigma[2])
        
        probs = probs / np.sum(probs)
        
        expected_counts += expected_photons_per_vertex * probs
    
    bin_volume = bin_width * bin_height * bin_depth
    expected_counts *= bin_volume
    
    return expected_counts


def distribution_to_pdf(mu, sigma, params, resolution_factor=4):
    x_grid = np.linspace(0, params.R, params.R * resolution_factor) + 1 / (2 * resolution_factor)
    y_grid = np.linspace(0, params.C, params.C * resolution_factor) + 1 / (2 * resolution_factor)
    z_grid = np.linspace(0, params.T, params.T) 
    Xc, Yc, Zc = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    x_term = ((Xc - mu[0])/sigma[0])**2
    y_term = ((Yc - mu[1])/sigma[1])**2
    z_term = ((Zc - mu[2])/sigma[2])**2

    pdf = np.exp(-0.5 * (x_term + y_term + z_term)) / ((2*np.pi)**(3/2) * sigma[0] * sigma[1] * sigma[2])
    
    return pdf

def main():
    params = Params()
    
    for i in range(5):
        vertex_positions, photon_counts = simulate_vertex_electron_photon_explicit(
            params=params,
            seed=123 + i
        )

        print('true vertex positions:', vertex_positions)

        x_grid = np.arange(params.R)
        y_grid = np.arange(params.C)
        z_grid = np.arange(params.T)
        # Combine electron and photon spread in quadrature for each dimension
        sigma = np.sqrt(params.se**2 + params.sp**2)
        
        mle_vertex_positions, mle_uncertainties, mle_log_likelihood = compute_mle_vertex_positions(
            photon_counts, params, sigma, max_iterations=2000, tolerance=1e-1, learning_rate=0.01
        )
        mle_pdf = np.sum([distribution_to_pdf(mu, sigma, params, resolution_factor=8) for mu, sigma in zip(mle_vertex_positions, mle_uncertainties)], axis=0)

        for j in range(params.N):
            print(f"Vertex {j} position: {vertex_positions[j]}")
            print(f"Vertex {j} uncertainty: {mle_uncertainties[j]}")
        
        max_photon = photon_counts.max() if photon_counts.size > 0 else 1
        # max_est = estimated_counts.max() if estimated_counts.size > 0 else 1
        
        hit1 = photon_counts / (max_photon if max_photon>0 else 1)
        # hit2 = estimated_counts / (max_est if max_est>0 else 1)
        hit2 = mle_pdf / (mle_pdf.max() if mle_pdf.size > 0 else 1)

        print('estimated:', mle_vertex_positions)
        print('true:', vertex_positions)

        gif_name = f"hit_comparison_event_{i+1}"
        # plot_hit_comparison(hit1, hit2, filename=gif_name, title1="Observed Photoelectrons", title2="MLE Uncertainty Distribution")

        plot_vertex_positions_comparison(vertex_positions, mle_vertex_positions, params, filename=f"vertex_positions_event_{i+1}")


if __name__ == "__main__":
    main()