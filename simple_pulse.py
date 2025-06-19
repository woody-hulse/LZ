import numpy as np
from dataclasses import dataclass, field

# Parameters
DIFFWIDTH       = 300   # ns
G2              = 50
EGASWIDTH       = 450   # ns
PHDWIDTH        = 10    # ns
SAMPLERATE      = 10    # ns
NUM_ELECTRONS   = 150

# Define simulation parameters
@dataclass
class Params:
    R: int = 24
    C: int = 24
    T: int = 700

    # N: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    E: float = NUM_ELECTRONS
    P: float = G2
    
    rv: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.4]))
    se: np.ndarray = field(default_factory=lambda: np.array([1, 1, DIFFWIDTH / SAMPLERATE]))
    sp: np.ndarray = field(default_factory=lambda: np.array([2, 2, EGASWIDTH / SAMPLERATE]))

def simulate_vertex_electron_photon(N_list: list[int], params: Params, seed: int = None):
    """
    Simulate a single realization of the vertex-electron-photon process:
    
    1) Generate N vertex positions in a 3D region [0..R, 0..C, 0..T].
       Vertices are uniformly distributed in the middle rv% of each dimension.
    
    2) For each vertex i:
         - total photon_i ~ Poisson(E * P)
         - each photon's position ~ Normal(vertex_pos_i, diag(se_x^2 + sp_x^2, se_y^2 + sp_y^2, se_z^2 + sp_z^2))
    
    3) Bin all photon positions into an RxCxT integer array (floor the photon coords
       to get indices, discard out-of-bounds).
    
    Parameters
    ----------
    params : Params
        Simulation parameters
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    vertex_positions : ndarray of shape (N, 3)
        The (x,y,z) positions of the vertices.

    photon_counts : ndarray of shape (R, C, T)
        The binned photon counts over the 3D grid.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate the bounds for uniform distribution in each dimension
    x_margin = params.R * (1 - params.rv[0]) / 2
    y_margin = params.C * (1 - params.rv[1]) / 2
    z_margin = params.T * (1 - params.rv[2]) / 2
    
    x_min, x_max = x_margin, params.R - x_margin
    y_min, y_max = y_margin, params.C - y_margin
    z_min, z_max = z_margin, params.T - z_margin
    
    # Generate vertex positions uniformly in the middle rv of each dimension
    N = np.random.choice(N_list)
    vertex_positions = np.zeros((N, 3))
    vertex_positions[:, 0] = np.random.uniform(x_min, x_max, N)
    vertex_positions[:, 1] = np.random.uniform(y_min, y_max, N)
    vertex_positions[:, 2] = np.random.uniform(z_min, z_max, N)
    
    # Prepare an RxCxT array for the photon counts
    photon_counts = np.zeros((params.R, params.C, params.T), dtype=int)
    
    # Effective sigma for photon offset from vertex (per dimension)
    sigma_photon = np.sqrt(params.se**2 + params.sp**2)

    # For each vertex, sample number of photon, sample photon offsets, bin them
    for i in range(N):
        n_photon = np.random.poisson(params.E * params.P)
        if n_photon == 0:
            continue
        
        photon_offsets = np.zeros((n_photon, 3))
        photon_offsets[:, 0] = sigma_photon[0] * np.random.randn(n_photon)
        photon_offsets[:, 1] = sigma_photon[1] * np.random.randn(n_photon)
        photon_offsets[:, 2] = sigma_photon[2] * np.random.randn(n_photon)
        
        photon_positions = vertex_positions[i] + photon_offsets
        
        # Convert continuous positions to voxel indices
        indices = np.floor(photon_positions).astype(int)
        
        # Keep only in-bounds photon
        valid = (
            (indices[:,0] >= 0) & (indices[:,0] < params.R) &
            (indices[:,1] >= 0) & (indices[:,1] < params.C) &
            (indices[:,2] >= 0) & (indices[:,2] < params.T)
        )
        indices = indices[valid]
        
        if len(indices) > 0:
            unique_idx, idx_counts = np.unique(indices, axis=0, return_counts=True)
            for (x, y, z), c in zip(unique_idx, idx_counts):
                photon_counts[x, y, z] += c

    return vertex_positions, photon_counts, N

def gaussian(x, C, mu, sigma):
    return C*np.exp(-(x-mu)**2/(2*sigma**2))

def simulate_vertex_electron_photon_explicit(N_list: list[int], params: Params, seed: int = None):
    """
    Perform the entire vertex->electron->photon sampling chain explicitly, but only
    store & return vertex positions + final photon counts.

    Steps:
      1) Generate N vertices in [0..R, 0..C, 0..T].
         Vertices are uniformly distributed in the middle rv of each dimension.

      2) For each vertex i:
         - electrons_i ~ Poisson(E)
         - each electron's position = vertex_pos_i + Normal(0, diag(se_x^2, se_y^2, se_z^2))
         - For each electron j:
             photon_j ~ Poisson(P)
             each photon's position = electron_position_j + Normal(0, diag(sp_x^2, sp_y^2, sp_z^2))
             binned into R x C x T

    Parameters
    ----------
    params : Params
        Simulation parameters
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    vertex_positions : ndarray of shape (N, 3)
        The (x, y, z) positions of the vertices.

    photon_counts : ndarray of shape (R, C, T)
        The final binned photon counts in the 3D grid.
    """

    if seed is not None:
        np.random.seed(seed)

    photon_interval_width = PHDWIDTH // SAMPLERATE * 3
    photon_interval = np.array([i for i in range(-photon_interval_width, photon_interval_width + 1)])
    phd_sample_width = PHDWIDTH / SAMPLERATE

    x_margin = params.R * (1 - params.rv[0]) / 2
    y_margin = params.C * (1 - params.rv[1]) / 2
    z_margin = params.T * (1 - params.rv[2]) / 2
    
    x_min, x_max = x_margin, params.R - x_margin
    y_min, y_max = y_margin, params.C - y_margin
    z_min, z_max = z_margin, params.T - z_margin
    
    N = np.random.choice(N_list)
    vertex_positions = np.zeros((N, 3))
    vertex_positions[:, 0] = np.random.uniform(x_min, x_max, N)
    vertex_positions[:, 1] = np.random.uniform(y_min, y_max, N)
    vertex_positions[:, 2] = np.random.uniform(z_min, z_max, N)

    pulse = np.zeros((params.R, params.C, params.T), dtype=np.float16)

    for i in range(N):
        vertex_pos = vertex_positions[i]

        num_electrons = np.random.poisson(params.E)
        if num_electrons == 0:
            continue

        electron_pulses = np.random.normal(vertex_pos[2], params.se[2], size=num_electrons)
        num_photons = np.random.poisson(params.P, size=num_electrons)
        
        er = np.random.normal(vertex_pos[0], params.se[0], size=num_electrons)
        ec = np.random.normal(vertex_pos[1], params.se[1], size=num_electrons)
        
        for e in range(num_electrons):
            eri, eci = min(params.R - 1, max(0, int(er[e]))), min(params.C - 1, max(0, int(ec[e])))
            
            pr = np.random.normal(er[e], params.sp[0], size=num_photons[e])
            pc = np.random.normal(ec[e], params.sp[1], size=num_photons[e])
            photon_arrival_times = np.random.normal(electron_pulses[e], params.sp[2], num_photons[e])
            
            for p in range(num_photons[e]):
                photon_index = int(photon_arrival_times[p])
                photon_indices = photon_index + photon_interval
                valid_indices = (photon_indices >= 0) & (photon_indices < params.T)
                
                if np.any(valid_indices):
                    photon_emission = np.exp(-0.5 * ((photon_indices[valid_indices] - photon_arrival_times[p]) / phd_sample_width)**2) # / np.sqrt(2 * np.pi)
                    
                    pri, pci = min(params.R - 1, max(0, int(pr[p]))), min(params.C - 1, max(0, int(pc[p])))
                    
                    if 0 <= pri < params.R and 0 <= pci < params.C:
                        pulse[pri][pci][photon_indices[valid_indices]] += photon_emission

    return vertex_positions, pulse, N


def simulate_vertex_electron_photon_explicit_batch(N_list: list[int], params: Params, batch_size: int = 32, seed: int = None):
    max_N = max(N_list)
    large_sigma = 4
    
    if seed is not None:
        np.random.seed(seed)

    x_margin = params.R * (1 - params.rv[0]) / 2
    y_margin = params.C * (1 - params.rv[1]) / 2
    z_margin = params.T * (1 - params.rv[2]) / 2
    
    x_min, x_max = x_margin, params.R - x_margin
    y_min, y_max = y_margin, params.C - y_margin
    z_min, z_max = z_margin, params.T - z_margin

    batch_vertex_positions_x = np.random.uniform(x_min, x_max, (batch_size, max_N))
    batch_vertex_positions_y = np.random.uniform(y_min, y_max, (batch_size, max_N))
    batch_vertex_positions_z = np.random.uniform(z_min, z_max, (batch_size, max_N))

    batch_photon_counts = np.zeros((batch_size, params.R, params.C, params.T, max_N), dtype=int)

    batch_electron_offsets_x = params.se[0] * np.random.randn(batch_size, params.E * large_sigma, max_N)
    batch_electron_offsets_y = params.se[1] * np.random.randn(batch_size, params.E * large_sigma, max_N)
    batch_electron_offsets_z = params.se[2] * np.random.randn(batch_size, params.E * large_sigma, max_N)

    batch_electron_positions_x = batch_vertex_positions_x + batch_electron_offsets_x
    batch_electron_positions_y = batch_vertex_positions_y + batch_electron_offsets_y
    batch_electron_positions_z = batch_vertex_positions_z + batch_electron_offsets_z

    batch_photon_offsets_x = params.sp[0] * np.random.randn(batch_size, params.E * large_sigma, params.P * large_sigma, max_N)
    batch_photon_offsets_y = params.sp[1] * np.random.randn(batch_size, params.E * large_sigma, params.P * large_sigma, max_N)
    batch_photon_offsets_z = params.sp[2] * np.random.randn(batch_size, params.E * large_sigma, params.P * large_sigma, max_N)

    batch_photon_positions_x = np.floor(batch_electron_positions_x + batch_photon_offsets_x).astype(int)
    batch_photon_positions_y = np.floor(batch_electron_positions_y + batch_photon_offsets_y).astype(int)
    batch_photon_positions_z = np.floor(batch_electron_positions_z + batch_photon_offsets_z).astype(int)

    batch_num_vertices = np.random.choice(N_list, batch_size)
    batch_num_electrons = np.random.poisson(params.E, (batch_size, max_N))
    batch_num_photons = np.random.poisson(params.P, (batch_size, params.E * large_sigma, max_N))

    batch_vertex_mask = np.zeros((batch_size, max_N), dtype=bool)
    for b in range(batch_size):
        batch_vertex_mask[b, :batch_num_vertices[b]] = True
    
    batch_electron_mask = np.zeros((batch_size, max_N), dtype=bool)
    for b in range(batch_size):
        for i in range(batch_num_vertices[b]):
            batch_electron_mask[b, i, :batch_num_electrons[b, i]] = True
    
    batch_photon_mask = np.zeros((batch_size, params.E * large_sigma, max_N), dtype=bool)
    for b in range(batch_size):
        for i in range(batch_num_vertices[b]):
            for j in range(batch_num_electrons[b, i]):
                batch_photon_mask[b, i, j, :batch_num_photons[b, i, j]] = True

    batch_photon_positions_x = batch_photon_positions_x[batch_vertex_mask & batch_electron_mask & batch_photon_mask]
    batch_photon_positions_y = batch_photon_positions_y[batch_vertex_mask & batch_electron_mask & batch_photon_mask]
    batch_photon_positions_z = batch_photon_positions_z[batch_vertex_mask & batch_electron_mask & batch_photon_mask]
    
    batch_photon_counts = np.zeros((batch_size, params.R, params.C, params.T), dtype=int)

    for b in range(batch_size):
        for i in range(batch_num_vertices[b]):
            for j in range(batch_num_electrons[b, i]):
                for k in range(batch_num_photons[b, i, j]):
                    batch_photon_counts[b, batch_photon_positions_x[b, i, j, k], batch_photon_positions_y[b, i, j, k], batch_photon_positions_z[b, i, j, k]] += 1

    batch_vertex_positions = np.concatenate(batch_vertex_positions_x, batch_vertex_positions_y, batch_vertex_positions_z, axis=0)

    return batch_vertex_positions, batch_photon_counts, batch_num_vertices


def vertex_electron_batch_generator(N_list: list[int], params: Params, batch_size: int = 128, steps_per_epoch: int = None, seed: int = None):
    while True:
        batch_seed = None if seed is None else seed + current_step + current_epoch * (steps_per_epoch or 0)
        
        vertex_positions_batch = []
        photon_counts_batch = np.zeros((batch_size, params.R, params.C, params.T), dtype=int)
        N_batch = np.zeros(batch_size, dtype=int)
        
        for b in range(batch_size):
            sample_seed = None if batch_seed is None else batch_seed + b
            vertex_positions, photon_counts, N = simulate_vertex_electron_photon_explicit(
                N_list=N_list,
                params=params,
                seed=sample_seed
            )
            
            vertex_positions_batch.append(vertex_positions)
            photon_counts_batch[b] = photon_counts
            N_batch[b] = N
        
        yield photon_counts_batch, vertex_positions_batch, N_batch


def simulate_fixed_vertex_electron(vertex_positions, electron_counts, params: Params, seed: int = None):
    """
    Generate a pulse using fixed vertex positions and fixed electron counts,
    simulating only the photon generation process.
    
    Parameters
    ----------
    vertex_positions : ndarray of shape (N, 3)
        The (x, y, z) positions of the vertices.
    electron_counts : ndarray of shape (N,)
        Number of electrons for each vertex.
    params : Params
        Simulation parameters.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    pulse : ndarray of shape (R, C, T)
        The generated pulse (photon counts in the 3D grid).
    photon_counts : list of lists
        Number of photons generated for each electron for each vertex.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    N = len(vertex_positions)
    
    # Set up pulse array and photon interval parameters
    pulse = np.zeros((params.R, params.C, params.T), dtype=np.float16)
    photon_interval_width = PHDWIDTH // SAMPLERATE * 3
    photon_interval = np.array([i for i in range(-photon_interval_width, photon_interval_width + 1)])
    phd_sample_width = PHDWIDTH / SAMPLERATE
    
    # Sample photon counts for each electron
    photon_counts = []
    for i in range(N):
        vertex_photons = []
        for e in range(electron_counts[i]):
            vertex_photons.append(np.random.poisson(params.P))
        photon_counts.append(vertex_photons)
    
    # Process each vertex
    for i in range(N):
        vertex_pos = vertex_positions[i]
        num_electrons = electron_counts[i]
        
        if num_electrons == 0:
            continue
            
        # Generate electron positions with Gaussian offset from vertex
        electron_pulses = np.random.normal(vertex_pos[2], params.se[2], size=num_electrons)
        er = np.random.normal(vertex_pos[0], params.se[0], size=num_electrons)
        ec = np.random.normal(vertex_pos[1], params.se[1], size=num_electrons)
        
        # Process each electron
        for e in range(num_electrons):
            eri, eci = min(params.R - 1, max(0, int(er[e]))), min(params.C - 1, max(0, int(ec[e])))
            
            num_photons_e = photon_counts[i][e]
            
            # Generate photon positions with Gaussian offset from electron
            pr = np.random.normal(er[e], params.sp[0], size=num_photons_e)
            pc = np.random.normal(ec[e], params.sp[1], size=num_photons_e)
            photon_arrival_times = np.random.normal(electron_pulses[e], params.sp[2], num_photons_e)
            
            # Process each photon
            for p in range(num_photons_e):
                photon_index = int(photon_arrival_times[p])
                photon_indices = photon_index + photon_interval
                valid_indices = (photon_indices >= 0) & (photon_indices < params.T)
                
                if np.any(valid_indices):
                    photon_emission = np.exp(-0.5 * ((photon_indices[valid_indices] - photon_arrival_times[p]) / phd_sample_width)**2) # / np.sqrt(2 * np.pi)
                    
                    pri, pci = min(params.R - 1, max(0, int(pr[p]))), min(params.C - 1, max(0, int(pc[p])))
                    
                    if 0 <= pri < params.R and 0 <= pci < params.C:
                        pulse[pri][pci][photon_indices[valid_indices]] += photon_emission
    
    return pulse, photon_counts




def generate_vertex_electron_params(N_list: list[int], params: Params, seed: int = None):
    """
    Generate vertex positions and electron counts according to simulation parameters,
    without performing the full photon generation simulation.
    
    Parameters
    ----------
    N_list : list[int]
        List of possible numbers of vertices.
    params : Params
        Simulation parameters.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    vertex_positions : ndarray of shape (N, 3)
        The (x, y, z) positions of the vertices.
    electron_counts : ndarray of shape (N,)
        Number of electrons for each vertex.
    N : int
        Number of vertices.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate the bounds for uniform distribution in each dimension
    x_margin = params.R * (1 - params.rv[0]) / 2
    y_margin = params.C * (1 - params.rv[1]) / 2
    z_margin = params.T * (1 - params.rv[2]) / 2
    
    x_min, x_max = x_margin, params.R - x_margin
    y_min, y_max = y_margin, params.C - y_margin
    z_min, z_max = z_margin, params.T - z_margin
    
    # Generate N vertices
    N = np.random.choice(N_list)
    vertex_positions = np.zeros((N, 3))
    vertex_positions[:, 0] = np.random.uniform(x_min, x_max, N)
    vertex_positions[:, 1] = np.random.uniform(y_min, y_max, N)
    vertex_positions[:, 2] = np.random.uniform(z_min, z_max, N)
    
    # Generate electron counts for each vertex
    electron_counts = np.random.poisson(params.E, size=N)
    
    return vertex_positions, electron_counts, N


if __name__ == "__main__":
    params = Params()
    
    vertex_positions, photon_counts_3d = simulate_vertex_electron_photon(
        params=params,
        seed=123
    )
    
    print("Simulated vertex positions (N=5):")
    print(vertex_positions)
    print("Shape of photon_counts_3d:", photon_counts_3d.shape)
    print("Total number of photon in the volume:", photon_counts_3d.sum())