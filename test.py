import numpy as np

# You already have (or can define) these functions:
#   1) model_expected_counts(x_grid, y_grid, z_grid, vertex_positions, sigma, E, P, ...)
#   2) poisson_log_likelihood(observed_counts, expected_counts)

def log_prior_uniform(vertex_positions, R, C, T):
    """
    A simple uniform prior for each vertex in [0,R]x[0,C]x[0,T].
    vertex_positions is shape (N, 3).
    Returns log(1/Volume^N) if all vertices are in-bounds, else -infinity.
    """
    # Check bounds
    if np.any(vertex_positions[:,0] < 0) or np.any(vertex_positions[:,0] > R):
        return -np.inf
    if np.any(vertex_positions[:,1] < 0) or np.any(vertex_positions[:,1] > C):
        return -np.inf
    if np.any(vertex_positions[:,2] < 0) or np.any(vertex_positions[:,2] > T):
        return -np.inf

    # If in-bounds, uniform means constant log-prior
    # The volume for 1 vertex is R*C*T, so for N vertices it's (R*C*T)^N.
    # log of that is N * log(R*C*T).
    # A constant log-prior does not affect relative posterior except for boundary cutoffs
    # so you can just return 0 if in-bounds, or -inf if out-of-bounds.
    # We return 0 for convenience.
    return 0.0

def log_posterior(vertex_positions, observed_photon, R, C, T, E, P, se, sp,
                  x_grid, y_grid, z_grid):
    """
    Posterior = prior * likelihood => log_posterior = log_prior + log_likelihood.
    
    vertex_positions: shape (N, 3).
    observed_photon: shape (R, C, T).
    """
    # 1) log-prior
    lp = log_prior_uniform(vertex_positions, R, C, T)
    if not np.isfinite(lp):
        return -np.inf  # out of bounds => posterior is 0 => log posterior is -inf
    
    # 2) log-likelihood
    sigma = np.sqrt(se**2 + sp**2)
    expected = model_expected_counts(
        x_grid, y_grid, z_grid,
        vertex_positions, sigma, E, P,
        bin_width=1.0, bin_height=1.0, bin_depth=1.0
    )
    ll = poisson_log_likelihood(observed_photon, expected)
    
    return lp + ll

def sample_vertex_positions_posterior(
    observed_photon, N,
    R, C, T,
    E, P, se, sp,
    n_samples=5000,
    step_size=0.5,
    seed=123
):
    """
    Simple Metropolis-Hastings sampler for the posterior distribution over vertex positions.

    Arguments
    ---------
    observed_photon : np.ndarray (R, C, T)
        Observed photon counts.
    N : int
        Number of vertices.
    R, C, T : int
        Dimensions of the bounding box.
    E, P : float
        Poisson means (vertex->electron, electron->photon).
    se, sp : float
        stdev for electron/photon positions.
    n_samples : int
        Number of MCMC samples to draw (not counting burn-in).
    step_size : float
        Standard deviation for the random-walk proposal around each vertex position dimension.
    seed : int
        Random seed.

    Returns
    -------
    samples : np.ndarray of shape (n_samples, N, 3)
        The chain of vertex-position samples.
        Each entry is one posterior sample (N vertex positions in 3D).
    log_posts : np.ndarray of shape (n_samples,)
        The log-posterior at each sample.
    
    Notes
    -----
    - This is a minimal random-walk Metropolis approach, suitable for demonstration.
    - For real use, consider:
        - burn-in / thinning
        - advanced methods (e.g. HMC, NUTS, PyMC, Stan)
        - multiple chains / diagnosing convergence
    """
    np.random.seed(seed)

    # 1) We define the 1D grids for x, y, z
    x_grid = np.arange(R)
    y_grid = np.arange(C)
    z_grid = np.arange(T)

    # 2) Initialize vertex positions randomly (uniform in bounding box)
    current_positions = np.column_stack([
        np.random.uniform(0, R, size=N),
        np.random.uniform(0, C, size=N),
        np.random.uniform(0, T, size=N)
    ])
    current_log_post = log_posterior(current_positions, observed_photon, R, C, T,
                                     E, P, se, sp, x_grid, y_grid, z_grid)

    # 3) Storage for the chain
    samples = np.zeros((n_samples, N, 3))
    log_posts = np.zeros(n_samples)

    # 4) Metropolis-Hastings sampling loop
    for i in range(n_samples):
        # Propose a new position by a Gaussian step around each vertex
        proposal = current_positions + step_size * np.random.randn(N, 3)
        
        # Evaluate log-posterior at the proposal
        proposal_log_post = log_posterior(proposal, observed_photon, R, C, T,
                                          E, P, se, sp, x_grid, y_grid, z_grid)
        
        # Accept/reject
        accept_ratio = np.exp(proposal_log_post - current_log_post)
        if np.random.rand() < accept_ratio:
            # Accept
            current_positions = proposal
            current_log_post = proposal_log_post
        
        # Store sample
        samples[i] = current_positions
        log_posts[i] = current_log_post

    return samples, log_posts


# ----------------------------------------------------------------
# Example Usage (Pseudo-code)
if __name__ == "__main__":
    # Suppose you have an observed event with shape (R, C, T)
    R, C, T = 20, 25, 15
    # e.g. from simulation
    # (for brevity, we won't re-show the entire simulation code here)
    observed_photon = np.zeros((R, C, T), dtype=int)

    # Known parameters
    N = 3
    E = 5.0
    P = 2.0
    se = 1.5
    sp = 0.5

    # Sample from the posterior
    samples, log_posts = sample_vertex_positions_posterior(
        observed_photon, N, R, C, T, E, P, se, sp,
        n_samples=2000,
        step_size=0.5,
        seed=42
    )
    
    # Inspect the final samples
    print("Shape of the posterior samples:", samples.shape)  # (2000, N, 3)
    print("Mean log-posterior:", log_posts.mean())
    
    # E.g. look at the last sample:
    print("Last sample vertex positions:\n", samples[-1])
    
    # You can also compute summary stats across the chain (after a burn-in)
    burn_in = 500
    chain_after_burn = samples[burn_in:]
    # e.g. mean position for each vertex
    vertex_means = chain_after_burn.mean(axis=0)  # shape (N, 3)
    vertex_stds = chain_after_burn.std(axis=0)    # shape (N, 3)
    print("Posterior mean vertex positions:\n", vertex_means)
    print("Posterior std dev vertex positions:\n", vertex_stds)
