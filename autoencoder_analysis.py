import pandas as pd
from scipy.ndimage import gaussian_filter1d

from autoencoder import *
from simple_pulse import simulate_vertex_electron_photon_explicit, Params, vertex_electron_batch_generator

def set_mpl_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "lines.markersize": 2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": True,
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    })


def plot_events(events, title='hit_pattern', subtitles=[]):
    assert len(events.shape) == 4, 'Events must be a 3D array with shape (num_events, num_rows * num_cols, num_samples)'
    events = np.transpose(events, axes=[3, 0, 1, 2])

    gif_frames = []
    for sample in tqdm(events):
        num_events = sample.shape[0]
        fig, ax = plt.subplots(1, num_events, figsize=(3*num_events, 3), dpi=100)
        fig.suptitle(title, fontsize=16)

        if num_events == 1: ax = [ax]

        for i, hit_pattern in enumerate(sample):
            ax[i].imshow(hit_pattern, vmin=0, vmax=5)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].grid(False)
            
            if len(subtitles) >= i + 1:
                ax[i].set_title(subtitles[i], fontsize=8)

        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.dpi
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_array = data.reshape(int(height), int(width), 4)

        image_frame = Image.fromarray(image_array)
        gif_frames.append(image_frame)
        plt.clf()
        plt.close()

    filename = re.sub(r'[^a-zA-Z0-9]', '', title.lower()) + '_gif.gif'
    gif_frames[0].save(
        filename,
        save_all = True,
        duration = 20,
        loop = 0,
        append_images = gif_frames[1:]
    )

    return filename

def plot_3d_scatter(event, title, save_path='', threshold=0.1, figsize=(12, 12), dpi=150, azimuth=30, elevation=30, cmap='viridis', t_group_size=1):
    
    event = np.clip(event, 0, 4)
    event = event.reshape((event.shape[0], event.shape[1], event.shape[2] // t_group_size, t_group_size))
    event = np.sum(event, axis=-1)

    norm_event = event / (np.max(event) + 1e-10)
    
    filled = norm_event > threshold

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    cmap_func = plt.get_cmap(cmap)
    
    # Create RGBA colors for voxels
    colors = np.zeros(filled.shape + (4,))
    for i in range(filled.shape[0]):
        for j in range(filled.shape[1]):
            for k in range(filled.shape[2]):
                if filled[i, j, k]:
                    value = round(norm_event[i, j, k] * 100) / 100
                    color = cmap_func(value)
                    colors[i, j, k] = [color[0], color[1], color[2], value]
    
    edge_colors = np.zeros(filled.shape + (4,))
    mask = np.where(filled)
    for i, j, k in zip(*mask):
        value = round(norm_event[i, j, k] * 100) / 100
        color = cmap_func(value)
        edge_colors[i, j, k] = [color[0], color[1], color[2], 0]
    
    if np.any(filled):
        alpha_values = np.unique(colors[filled, 3])
        
        for alpha in tqdm(alpha_values, desc="Plotting by alpha groups"):
            alpha_mask = np.zeros_like(filled, dtype=bool)
            alpha_mask_indices = np.where(filled & (np.abs(colors[:, :, :, 3] - alpha) < 1e-6))
            alpha_mask[alpha_mask_indices] = True
            
            if np.any(alpha_mask):
                group_voxels = np.zeros_like(event, dtype=bool)
                group_voxels[alpha_mask] = True
                
                group_facecolors = np.zeros((*event.shape, 4))
                group_edgecolors = np.zeros((*event.shape, 4))
                
                for i, j, k in zip(*alpha_mask_indices):
                    group_facecolors[i, j, k] = colors[i, j, k]
                    group_edgecolors[i, j, k] = edge_colors[i, j, k]
                
                ax.voxels(group_voxels, facecolors=group_facecolors, edgecolors=group_edgecolors)
    else:
        print("No voxels to plot")
        
    ax.set_xlabel('X', fontsize=16, labelpad=15)
    ax.set_ylabel('Y', fontsize=16, labelpad=15)
    ax.set_zlabel('Z (Time, ns)', fontsize=16, labelpad=15)
    # ax.set_title(title, fontsize=20)

    z_max = event.shape[2]
    z_start = int(z_max * 0.1)
    z_end = int(z_max * 0.9)
    ax.set_zlim(z_start, z_end)
    
    z_ticks = np.arange(z_start, z_end, 1).astype(int)
    z_tick_labels = [str(z * t_group_size * SAMPLERATE) if z % 10 == 0 else '' for z in z_ticks]
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_tick_labels)
    ax.set_box_aspect((event.shape[0], event.shape[1], event.shape[2] * 0.8))
    
    ax.view_init(elevation, azimuth)

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.grid(False)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0

    total_phd = np.sum(event)
    max_phd = np.max(event)
    active_voxels = np.sum(filled)
    
    # stats_text = f"Total PHD: {total_phd:.1f}\nMax PHD: {max_phd:.1f}\nActive Voxels: {active_voxels}"
    # ax.text2D(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=16,
    #           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    x_max, y_max, z_max = event.shape
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    full_path = os.path.join(save_path, title + '3d_voxels.png')
    plt.savefig(full_path, bbox_inches='tight')
    
    # plt.tight_layout()
    # plt.show()
    plt.clf()
    
    return fig


def plot_3d_scatter_with_profiles(event, title, save_path='data/', threshold=0.1, figsize=(12, 8), dpi=150, azimuth=30, elevation=30, cmap='viridis', t_group_size=1, return_fig=False, normalize=True):
    
    event = np.clip(event, 0, 4)
    event = event.reshape((event.shape[0], event.shape[1], event.shape[2] // t_group_size, t_group_size))
    event = np.sum(event, axis=-1)

    if normalize:
        norm_event = event / (np.max(event) + 1e-10)
    else:
        norm_event = event
    
    filled = norm_event > threshold

    x_profile = np.sum(event, axis=(1, 2))
    y_profile = np.sum(event, axis=(0, 2))
    z_profile = np.sum(event, axis=(0, 1))
    
    if np.max(x_profile) > 0:
        x_profile = x_profile / np.max(x_profile) * event.shape[0] * 0.4
    if np.max(y_profile) > 0:
        y_profile = y_profile / np.max(y_profile) * event.shape[1] * 0.4
    if np.max(z_profile) > 0:
        z_profile = z_profile / np.max(z_profile) * event.shape[2] * 0.4
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    cmap_func = plt.get_cmap(cmap)
    
    colors = np.zeros(filled.shape + (4,))
    for i in range(filled.shape[0]):
        for j in range(filled.shape[1]):
            for k in range(filled.shape[2]):
                if filled[i, j, k]:
                    colors[i, j, k] = cmap_func(norm_event[i, j, k])
    
    indices = np.where(filled)
    if len(indices[0]) > 0:
        sizes = norm_event[indices] * 100
        ax.scatter(
            indices[0], indices[1], indices[2],
            c=colors[indices],
            s=sizes,
            alpha=0.7
        )
    
    x_indices = np.arange(len(x_profile))
    y_pos = event.shape[1]
    z_pos = event.shape[2] * 0.1
    
    x_indices_high_res = np.linspace(0, len(x_profile)-1, len(x_profile) * 5)
    x_profile_interp = np.interp(x_indices_high_res, np.arange(len(x_profile)), x_profile)
    smoothed_x_profile = gaussian_filter1d(x_profile_interp, sigma=1.5)
    
    x_grid, z_grid = np.meshgrid(x_indices_high_res, [z_pos, z_pos])
    y_grid = np.zeros_like(x_grid)
    y_grid[0] = y_pos
    y_grid[1] = y_pos - smoothed_x_profile / 2
    
    ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.5, shade=True, antialiased=True, edgecolor='none')
    
    y_indices = np.arange(len(y_profile))
    x_pos = event.shape[0]
    z_pos = event.shape[2] * 0.1
    
    y_indices_high_res = np.linspace(0, len(y_profile)-1, len(y_profile) * 5)
    y_profile_interp = np.interp(y_indices_high_res, np.arange(len(y_profile)), y_profile)
    smoothed_y_profile = gaussian_filter1d(y_profile_interp, sigma=1.5)
    
    y_grid, z_grid = np.meshgrid(y_indices_high_res, [z_pos, z_pos])
    x_grid = np.zeros_like(y_grid)
    x_grid[0] = x_pos
    x_grid[1] = x_pos - smoothed_y_profile / 2
    
    ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.5, shade=True, antialiased=True, edgecolor='none')
    
    z_max = event.shape[2]
    z_start = int(z_max * 0.1)
    z_end = int(z_max * 0.9)
    z_indices = np.arange(z_start, z_end)
    x_pos = event.shape[0]
    y_pos = 0

    z_indices_high_res = np.linspace(z_indices[0], z_indices[-1], len(z_indices) * 5)
    z_profile_interp = np.interp(z_indices_high_res, z_indices, z_profile[z_indices])
    smoothed_profile = gaussian_filter1d(z_profile_interp, sigma=1.5)

    xy_profile = np.sum(event, axis=2)
    xy_profile = xy_profile / (np.max(xy_profile) + 1e-10)

    z_level = z_max * 0.9
    x_range = np.arange(event.shape[0])
    y_range = np.arange(event.shape[1])
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = np.ones_like(x_grid) * z_level
    
    xy_profile_plot = xy_profile.T
    surf = ax.plot_surface(
        x_grid, y_grid, z_grid, 
        facecolors=plt.cm.get_cmap(cmap)(xy_profile_plot),
        alpha=0.7,
        shade=False,
        antialiased=True
    )
    
    z_grid, dummy = np.meshgrid(z_indices_high_res, [0, 1])
    x_grid = np.zeros_like(z_grid)
    x_grid[0] = x_pos
    x_grid[1] = x_pos - smoothed_profile / 2
    y_grid = np.zeros_like(z_grid)
    y_grid[:] = y_pos
    ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.5, shade=True, antialiased=True, edgecolor='none')

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    z_max = event.shape[2]
    z_start = int(z_max * 0.1)
    z_end = int(z_max * 0.9)
    ax.set_zlim(z_start, z_end)
    
    z_ticks = np.arange(z_start, z_end, 1).astype(int)
    z_tick_labels = [str(z * t_group_size * SAMPLERATE) if z % 10 == 0 else '' for z in z_ticks]
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_tick_labels)
    ax.set_box_aspect((event.shape[0], event.shape[1], event.shape[2] * 0.8))

    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z (ns)', fontsize=16)
    
    ax.set_xlim(0, event.shape[0])
    ax.set_ylim(0, event.shape[1])
    ax.set_zlim(event.shape[2] * 0.1, event.shape[2] * 0.9)
    
    ax.view_init(elev=elevation, azim=azimuth)
    
    full_path = os.path.join(save_path, title + '3d_voxels_with_profiles.png')
    plt.savefig(full_path, bbox_inches='tight')

    if return_fig:
        return fig
    else:
        plt.close()

def plot_3d_multi_view(event, title="3D Event Multi-View", save_path='', threshold=0.001, 
                    figsize=(16, 12), dpi=150, cmap='viridis'):
    """
    Create a multi-view visualization of 3D data with different projections.
    Displays 2D summed projections along each axis with square visualization.
    """
    event = np.clip(event, 0, None)
    
    # Calculate projections (sums along each axis)
    proj_xy = np.sum(event, axis=2)  # Z-projection (top view)
    proj_xz = np.sum(event, axis=1)  # Y-projection (side view)
    proj_yz = np.sum(event, axis=0)  # X-projection (front view)
    
    # Create figure with square subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=20)
    
    # Plot each projection with square aspect
    # Use square extent for all plots regardless of data dimensions
    im1 = axes[0, 0].imshow(proj_xy.T, origin='lower', cmap=cmap,
                          extent=[0, 1, 0, 1])  # Square extent
    axes[0, 0].set_title("XY Projection (Top View)", fontsize=16)
    axes[0, 0].set_xlabel(f"X [{event.shape[0]}]", fontsize=14)
    axes[0, 0].set_ylabel(f"Y [{event.shape[1]}]", fontsize=14)
    
    im2 = axes[0, 1].imshow(proj_xz.T, origin='lower', cmap=cmap,
                          extent=[0, 1, 0, 1])  # Square extent
    axes[0, 1].set_title("XZ Projection (Side View)", fontsize=16)
    axes[0, 1].set_xlabel(f"X [{event.shape[0]}]", fontsize=14)
    axes[0, 1].set_ylabel(f"Z [{event.shape[2]}]", fontsize=14)
    
    im3 = axes[1, 0].imshow(proj_yz.T, origin='lower', cmap=cmap,
                          extent=[0, 1, 0, 1])  # Square extent
    axes[1, 0].set_title("YZ Projection (Front View)", fontsize=16)
    axes[1, 0].set_xlabel(f"Y [{event.shape[1]}]", fontsize=14)
    axes[1, 0].set_ylabel(f"Z [{event.shape[2]}]", fontsize=14)
    
    # Remove ticks since coordinates are normalized
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticklabels(['0', 'mid', f'{ax.get_xlabel().split("[")[1][:-1]}'])
        ax.set_yticklabels(['0', 'mid', f'{ax.get_ylabel().split("[")[1][:-1]}'])
    
    # Add stats to the empty subplot
    stats_ax = axes[1, 1]
    stats_ax.axis('off')
    
    total_intensity = np.sum(event)
    max_intensity = np.max(event)
    active_voxels = np.sum(event > threshold)
    
    stats_text = (
        f"Statistics:\n\n"
        f"Total Intensity: {total_intensity:.1f}\n"
        f"Max Intensity: {max_intensity:.1f}\n"
        f"Active Voxels: {active_voxels}\n"
        f"Dimensions: {event.shape[0]} × {event.shape[1]} × {event.shape[2]}"
    )
    
    stats_ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=14,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label('Summed Intensity', fontsize=14)
    
    full_path = os.path.join(save_path, title.replace(" ", "_") + '_projections.png')
    plt.savefig(full_path, bbox_inches='tight')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    # plt.show()
    plt.clf()
    
    return fig

def plot_3d_comparison_voxels(raw_event, reconstructed_event, title="Raw vs Reconstructed Event", 
                              save_path='', threshold=0.001, figsize=(20, 10), dpi=150):
    """
    Create a side-by-side comparison of raw and reconstructed 3D events using voxels
    with transparent edges.
    """
    # Normalize events
    norm_raw = raw_event / (np.max(raw_event) + 1e-10)
    norm_recon = reconstructed_event / (np.max(reconstructed_event) + 1e-10)
    
    # Create boolean arrays for filled voxels
    filled_raw = norm_raw > threshold
    filled_recon = norm_recon > threshold
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=20, y=0.95)
    
    # Get colormap function
    cmap_func = plt.get_cmap('viridis')
    
    # Raw event plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Create RGBA colors for raw voxels
    colors_raw = np.zeros(filled_raw.shape + (4,))
    edge_colors_raw = np.zeros(filled_raw.shape + (4,))
    
    for i in range(filled_raw.shape[0]):
        for j in range(filled_raw.shape[1]):
            for k in range(filled_raw.shape[2]):
                if filled_raw[i, j, k]:
                    value = norm_raw[i, j, k]
                    color = cmap_func(value)
                    colors_raw[i, j, k] = [color[0], color[1], color[2], value]
                    edge_colors_raw[i, j, k] = [0, 0, 0, 0]  # Transparent edges
    
    # Plot raw voxels with transparent edges
    ax1.voxels(filled_raw, facecolors=colors_raw, edgecolors=edge_colors_raw)
    
    ax1.set_xlabel('X', fontsize=14)
    ax1.set_ylabel('Y', fontsize=14)
    ax1.set_zlabel('Z (Time)', fontsize=14)
    ax1.set_title('Raw Event', fontsize=16)
    
    # Remove ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.grid(False)
    
    # Add raw event statistics
    total_phd_raw = np.sum(raw_event)
    max_phd_raw = np.max(raw_event)
    active_voxels_raw = np.sum(filled_raw)
    
    ax1.text2D(0.05, 0.95, f"Total PHD: {total_phd_raw:.1f}\nMax PHD: {max_phd_raw:.1f}\nActive Voxels: {active_voxels_raw}", 
               transform=ax1.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Reconstructed event plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Create RGBA colors for reconstructed voxels
    colors_recon = np.zeros(filled_recon.shape + (4,))
    edge_colors_recon = np.zeros(filled_recon.shape + (4,))
    
    for i in range(filled_recon.shape[0]):
        for j in range(filled_recon.shape[1]):
            for k in range(filled_recon.shape[2]):
                if filled_recon[i, j, k]:
                    value = norm_recon[i, j, k]
                    color = cmap_func(value)
                    colors_recon[i, j, k] = [color[0], color[1], color[2], value]
                    edge_colors_recon[i, j, k] = [0, 0, 0, 0]  # Transparent edges
    
    # Plot reconstructed voxels with transparent edges
    ax2.voxels(filled_recon, facecolors=colors_recon, edgecolors=edge_colors_recon)
    
    ax2.set_xlabel('X', fontsize=14)
    ax2.set_ylabel('Y', fontsize=14)
    ax2.set_zlabel('Z (Time)', fontsize=14)
    ax2.set_title('Reconstructed Event', fontsize=16)
    
    # Remove ticks
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.grid(False)
    
    # Add reconstructed event statistics
    total_phd_recon = np.sum(reconstructed_event)
    max_phd_recon = np.max(reconstructed_event)
    active_voxels_recon = np.sum(filled_recon)
    phd_error = (total_phd_recon - total_phd_raw) / total_phd_raw * 100
    
    ax2.text2D(0.05, 0.95, 
               f"Total PHD: {total_phd_recon:.1f}\nMax PHD: {max_phd_recon:.1f}\nActive Voxels: {active_voxels_recon}\nPHD Error: {phd_error:.1f}%", 
               transform=ax2.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_func)
    sm.set_array([0, 1])
    cbar = plt.colorbar(sm, ax=[ax1, ax2], shrink=0.6)
    cbar.set_label('Normalized Intensity', fontsize=14)
    
    # Set the same view angles
    ax1.view_init(elevation=30, azimuth=30)
    ax2.view_init(elevation=30, azimuth=30)
    
    # Set axes limits
    x_max, y_max, z_max = raw_event.shape
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, y_max)
    ax1.set_zlim(0, z_max)
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, y_max)
    ax2.set_zlim(0, z_max)
    
    # Save figure
    full_path = os.path.join(save_path, title.replace(" ", "_") + '_3d_voxels_comparison.png')
    plt.savefig(full_path, bbox_inches='tight')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.show()
    plt.clf()

def vis_latent_space_categories(data_categories, data_categories_labels, title):
    # plt.rcParams['figure.dpi'] = 120
    
    for data, label in zip(data_categories, data_categories_labels):
        plt.scatter(data[:, 0], data[:, 1], label=label, s=0.5)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.legend()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(re.sub(r'[^a-zA-Z0-9]', '', title.lower()) + '.png')
    # plt.show()
    plt.clf()
    
def vis_latent_space_gradients(latent_space, labels, title, colorbar_label):
    # plt.rcParams['figure.dpi'] = 120
    
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels, cmap='viridis', s=0.5)
    plt.title(title)
    plt.colorbar().set_label(colorbar_label)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(re.sub(r'[^a-zA-Z0-9]', '', title.lower()) + '.png')
    # plt.show()
    plt.clf()
    

def vis_latent_space_num_scatters(fit_model, latent_space, Y, N=4):
    num_scatters = np.where(Y > 0, 1, 0).sum(axis=(1, 2))
    
    num_sactters_categories = [latent_space[np.where(num_scatters == i)] for i in range(1, N + 1)]
    num_scatters_labels = [f'{i} scatter' + ('s' if i > 1 else '') for i in range(1, N + 1)]
    
    vis_latent_space_categories(num_sactters_categories, num_scatters_labels, f'Autoencoder Latent Space by Number of Scatters {type(fit_model).__name__.upper()}')


def vis_latent_space_phd(fit_model, latent_space, XC):
    phd = XC.sum(axis=(1, 2))
    vis_latent_space_gradients(latent_space, phd, f'Autoencoder Latent Space by Total Photoelectrons Deposited {type(fit_model).__name__.upper()}', colorbar_label='phd')
    
def vis_latent_space_footprint(fit_model, latent_space, XC):
    footprint = np.where(XC.sum(axis=-1) > 4, 1, 0).sum(axis=-1)
    
    vis_latent_space_gradients(latent_space, footprint, f'Autoencoder Latent Space by Footprint Size {type(fit_model).__name__.upper()}', colorbar_label='Total # PMT')
    

def codebook_usage_histogram(vqvae, XC):
    indices = vqvae.encode_to_indices_probabilistic(XC).numpy().flatten().astype(int)
    codebook_usage = np.bincount(indices, minlength=vqvae.num_embeddings)
    codebook_usage_sorted = codebook_usage[np.argsort(codebook_usage)[::-1]] / codebook_usage.sum()

    # plt.rcParams['figure.dpi'] = 120
    
    plt.fill_between(np.arange(len(codebook_usage_sorted)), codebook_usage_sorted, color='blue', alpha=0.3)
    plt.plot(np.arange(len(codebook_usage_sorted)), codebook_usage_sorted, color='blue', label='Usage')

    plt.xlabel('Codebook Index')
    plt.ylabel('Usage (PDF)')
    plt.title('VQ-VAE Codebook Usage Distribution')
    plt.margins(0)
    plt.savefig('codebook_usage.png')
    # plt.show()
    plt.clf()


def get_encoded_data_generator(compression_func, data_generator):
    while True:
        XC, XYZ, P = next(iter(data_generator))
        XC_encoded = compression_func(XC)
        yield XC_encoded, XYZ, P

def run_aux_task(models, compression_funcs, labels, data_generator, val_data_generator, fname_suffix=''):
    def precompute_batches(generator, steps):
        return [next(generator) for _ in tqdm(range(steps))]

    def train_epoch(model, train_batches, desc=''):
        total_loss = 0.0
        for batch in tqdm(train_batches, desc=desc, ncols=100):
            x, y, _ = batch
            loss = model.step(x, y, training=True)
            if isinstance(loss, (list, tuple)):
                loss = loss[0]
            total_loss += loss
        return total_loss

    def test_epoch(model, val_batches):
        total_loss = 0.0
        for batch in val_batches:
            x, y, _ = batch
            loss = model.step(x, y, training=False)
            if isinstance(loss, (list, tuple)):
                loss = loss[0]
            total_loss += loss
        return total_loss
    
    def train_and_plot_model(model, compression_func, data_generator, val_data_generator, epochs=10, steps_per_epoch=64, val_steps=4):
        if compression_func is not None:
            data_generator = get_encoded_data_generator(compression_func, data_generator)
            val_data_generator = get_encoded_data_generator(compression_func, val_data_generator)
        
        initial_val_loss = test_epoch(model, precompute_batches(val_data_generator, val_steps)) / val_steps
        
        train_times, val_losses = [0], [initial_val_loss]
        
        for epoch in range(epochs):
            train_batches = precompute_batches(data_generator, steps_per_epoch)
            val_batches = precompute_batches(data_generator, val_steps)
            
            desc = f'epoch {epoch + 1}/{epochs} ({model.name})'
            t0 = time.time()
            loss = train_epoch(model, train_batches, desc=desc)
            train_time = time.time() - t0
            avg_loss = loss / steps_per_epoch
            avg_val_loss = test_epoch(model, val_batches) / val_steps
            print(desc + f' - loss: {avg_loss:.3f}, val_loss: {avg_val_loss:.3f}')
            
            train_times.append(train_time)
            val_losses.append(avg_val_loss)
                
        K.clear_session()
            
        cdf_train_times = [sum(train_times[:i + 1]) for i in range(len(train_times))]
        best_val_losses = [min(val_losses[:i + 1]) for i in range(len(val_losses))]
            
        return cdf_train_times, best_val_losses
    
    epochs = 50
    steps_per_epoch = 64
    val_steps = 8
            
    model_cdf_train_times, model_best_val_losses = [], []
    
    for model, compression_func in zip(models, compression_funcs):
        cdf_train_times, best_val_losses = train_and_plot_model(model, compression_func, data_generator, val_data_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, val_steps=val_steps)
        model_cdf_train_times.append(cdf_train_times)
        model_best_val_losses.append(best_val_losses)
    
    sample_batch, _, _ = next(iter(data_generator))
    batch_size = sample_batch.shape[0]
    
    plt.figure()
    epochs_axis = np.arange(0, epochs + 1) * steps_per_epoch * batch_size
    for label, val_losses in zip(labels, model_best_val_losses):
        plt.plot(epochs_axis, val_losses, '-o', label=label, markersize=2)
    plt.xlabel('Training Samples')
    plt.ylabel('Validation Loss')
    plt.title('Sample Efficiency: Validation Loss (Best) vs Training Samples')
    plt.legend()
    plt.savefig(f'raw_vs_compressed_sample_efficiency{fname_suffix}.png')
    
    plt.figure()
    for label, cdf_train_times, val_losses in zip(labels, model_cdf_train_times, model_best_val_losses):
        plt.plot(cdf_train_times, val_losses, '-o', label=label, markersize=2)
    plt.xlabel('Training Time (s)')
    plt.ylabel('Validation Loss')
    plt.title('Time Efficiency: Validation Loss (Best) vs Training Time')
    plt.legend()
    plt.savefig(f'raw_vs_compressed_time_efficiency{fname_suffix}.png')


def compute_mle_for_sample(N, i, raw_data, autoencoder_reconstruction, params, sigma):
    max_iterations = 3000
    tolerance = 1e-1
    learning_rate = 0.01

    positions, _, _ = compute_mle_vertex_positions(
        N, raw_data[i], params, sigma, max_iterations=max_iterations, tolerance=tolerance, learning_rate=learning_rate
    )
    recon_positions, _, _ = compute_mle_vertex_positions(
        N, autoencoder_reconstruction[i], params, sigma, max_iterations=max_iterations, tolerance=tolerance, learning_rate=learning_rate
    )
    return i, positions, recon_positions

def likelihood_test(model, samples=1024, tune=False, loss=None, use_multiprocessing=False):
    params = Params()
    batch_size = 128
    data_generator = vertex_electron_batch_generator([1, 2, 3, 4], params, batch_size)

    if tune:
        def autoencoder_generator_from_generator(generator):
            while True:
                x, y, N = next(iter(generator))
                x = np.array(np.reshape(x, (batch_size, params.R * params.C, params.T)), dtype=np.float32)
                yield x, x

        if loss is None:
            print('ERROR: if tuning, please provide a loss')
            return

        autoencoder_data_generator = autoencoder_generator_from_generator(data_generator)
        validation_autoencoder_data_generator = autoencoder_generator_from_generator(data_generator)
        train_models(
            [model], [loss], [tf.keras.optimizers.Adam(3e-4)], 
            autoencoder_data_generator, validation_autoencoder_data_generator, 
            epochs=0, steps_per_epoch=64, batch_size=batch_size, 
            use_checkpoints=True,
            ckpt_dir=f'ckpts/likelihood_tuning_'
        )

    N = 3
    all_vertex_positions = np.empty((samples * N, 3))
    all_mle_vertex_positions = np.empty((samples * N, 3))
    all_reconstruction_vertex_positions = np.empty((samples * N, 3))
    
    pool = None
    if use_multiprocessing:
        num_processes = mp.cpu_count()
        pool = mp.Pool(processes=num_processes)
    
    data_generator = vertex_electron_batch_generator([N], params, batch_size)
    for batch in range(0, samples, batch_size):
        current_batch_size = min(batch_size, samples - batch)
        raw_data, vertex_positions, _ = next(iter(data_generator))
        raw_data = raw_data[:current_batch_size]
        vertex_positions = vertex_positions[:current_batch_size]
        raw_data_ = np.array(np.reshape(raw_data, (current_batch_size, params.R * params.C, params.T)), dtype=np.float32)
        autoencoder_reconstruction_ = model(raw_data_)
        if isinstance(autoencoder_reconstruction_, (list, tuple)):
            autoencoder_reconstruction_ = autoencoder_reconstruction_[0]
        autoencoder_reconstruction = np.reshape(autoencoder_reconstruction_, (current_batch_size, params.R, params.C, params.T))

        sigma = np.sqrt(params.se**2 + params.sp**2)
        raw_mle_vertex_positions = np.empty((current_batch_size, N, 3))
        reconstruction_mle_vertex_positions = np.empty((current_batch_size, N, 3))
        
        if use_multiprocessing:
            compute_func = partial(N, compute_mle_for_sample, 
                                raw_data=raw_data, 
                                autoencoder_reconstruction=autoencoder_reconstruction, 
                                params=params, 
                                sigma=sigma)
            
            results = list(tqdm(
                pool.imap(compute_func, range(current_batch_size)),
                total=current_batch_size,
                desc='computing mle',
                ncols=120
            ))
            
            # Process the results
            for i, raw_positions, recon_positions in results:
                raw_mle_vertex_positions[i] = raw_positions
                reconstruction_mle_vertex_positions[i] = recon_positions
        else:
            # Sequential processing
            for i in tqdm(range(current_batch_size), desc='computing mle', ncols=120):
                _, raw_positions, recon_positions = compute_mle_for_sample(
                    N, i, raw_data, autoencoder_reconstruction, params, sigma
                )
                raw_mle_vertex_positions[i] = raw_positions
                reconstruction_mle_vertex_positions[i] = recon_positions
        
        start_idx = batch * N
        end_idx = start_idx + current_batch_size * N
        
        # Permute MLE and reconstruction positions to best match true vertex positions
        for i in range(current_batch_size):
            for j in range(N):
                idx = start_idx + i * N + j
                true_pos = vertex_positions[i][j]
                mle_distances = np.sqrt(np.sum((raw_mle_vertex_positions[i] - true_pos)**2, axis=1))
                recon_mle_distances = np.sqrt(np.sum((reconstruction_mle_vertex_positions[i] - true_pos)**2, axis=1))
                
                all_vertex_positions[idx] = true_pos
                all_mle_vertex_positions[idx] = raw_mle_vertex_positions[i, np.argmin(mle_distances)]
                all_reconstruction_vertex_positions[idx] = reconstruction_mle_vertex_positions[i, np.argmin(recon_mle_distances)]
    
    if use_multiprocessing:
        pool.close()
        pool.join()
    
    # Calculate errors between true positions and MLE/reconstruction positions
    true_vs_mle_errors = np.sqrt(np.sum((all_vertex_positions - all_mle_vertex_positions)**2, axis=1))
    true_vs_recon_errors = np.sqrt(np.sum((all_vertex_positions - all_reconstruction_vertex_positions)**2, axis=1))
    
    component_errors_mle = all_vertex_positions - all_mle_vertex_positions
    component_errors_recon = all_vertex_positions - all_reconstruction_vertex_positions
    
    plots_dir = f"likelihood_analysis_{model.name}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculate summary statistics
    mle_mean = np.mean(true_vs_mle_errors)
    mle_median = np.median(true_vs_mle_errors)
    mle_std = np.std(true_vs_mle_errors)
    mle_95pct = np.percentile(true_vs_mle_errors, 95)
    mle_max = np.max(true_vs_mle_errors)
    
    recon_mean = np.mean(true_vs_recon_errors)
    recon_median = np.median(true_vs_recon_errors)
    recon_std = np.std(true_vs_recon_errors)
    recon_95pct = np.percentile(true_vs_recon_errors, 95)
    recon_max = np.max(true_vs_recon_errors)
    
    improvement_pct = (mle_mean - recon_mean) / mle_mean * 100
    
    print("\nSummary Statistics:")
    print(f"MLE Mean Error: {mle_mean:.4f}")
    print(f"Reconstruction Mean Error: {recon_mean:.4f}")
    print(f"Improvement: {improvement_pct:.2f}%")
    print(f"MLE Median Error: {mle_median:.4f}")
    print(f"Reconstruction Median Error: {recon_median:.4f}")
    print(f"MLE 95th Percentile Error: {mle_95pct:.4f}")
    print(f"Reconstruction 95th Percentile Error: {recon_95pct:.4f}")
    
    # Create and save visualizations
    plot_error_distribution_kde(true_vs_mle_errors, true_vs_recon_errors, mle_mean, recon_mean, plots_dir)
    plot_cumulative_error_distribution(true_vs_mle_errors, true_vs_recon_errors, plots_dir)
    plot_component_errors_violin(component_errors_mle, component_errors_recon, plots_dir)
    plot_spatial_error_distribution(all_vertex_positions, true_vs_mle_errors, true_vs_recon_errors, plots_dir)
    plot_error_vs_distance(all_vertex_positions, true_vs_mle_errors, true_vs_recon_errors, params, plots_dir)
    plot_error_correlation(true_vs_mle_errors, true_vs_recon_errors, plots_dir)
    plot_per_vertex_errors(N, true_vs_mle_errors, true_vs_recon_errors, params, plots_dir)
    plot_radial_angular_errors(all_vertex_positions, all_mle_vertex_positions, all_reconstruction_vertex_positions, plots_dir)
    plot_bootstrap_analysis(true_vs_mle_errors, true_vs_recon_errors, mle_mean, recon_mean, plots_dir)
    
    create_summary_report(model.name, mle_mean, recon_mean, improvement_pct, mle_median, recon_median, 
                         mle_std, recon_std, mle_95pct, recon_95pct, component_errors_mle, 
                         component_errors_recon, plots_dir)
    
    print(f"\nAnalysis complete. Visualizations and report saved to: {plots_dir}/")
    
    return {
        'true_positions': all_vertex_positions,
        'mle_positions': all_mle_vertex_positions,
        'reconstruction_positions': all_reconstruction_vertex_positions,
        'mle_errors': true_vs_mle_errors,
        'reconstruction_errors': true_vs_recon_errors,
        'plots_dir': plots_dir
    }

def plot_error_distribution_kde(true_vs_mle_errors, true_vs_recon_errors, mle_mean, recon_mean, plots_dir):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(true_vs_mle_errors, label='Raw MLE Error', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(true_vs_recon_errors, label='Reconstruction MLE Error', fill=True, alpha=0.3, linewidth=2)
    plt.xlabel('Error Distance', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Error Distribution Comparison', fontsize=16)
    plt.axvline(mle_mean, color='blue', linestyle='--', label=f'Raw MLE Mean: {mle_mean:.4f}')
    plt.axvline(recon_mean, color='orange', linestyle='--', label=f'Reconstruction MLE Mean: {recon_mean:.4f}')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/error_distribution_kde.png", dpi=300)
    plt.show()

def plot_cumulative_error_distribution(true_vs_mle_errors, true_vs_recon_errors, plots_dir):
    plt.figure(figsize=(10, 6))
    sorted_mle_errors = np.sort(true_vs_mle_errors)
    sorted_recon_errors = np.sort(true_vs_recon_errors)
    p = np.linspace(0, 100, len(sorted_mle_errors))
    plt.plot(sorted_mle_errors, p, label='Raw MLE Error', linewidth=2)
    plt.plot(sorted_recon_errors, p, label='Reconstruction MLE Error', linewidth=2)
    plt.xlabel('Error Distance', fontsize=14)
    plt.ylabel('Percentile', fontsize=14)
    plt.title('Cumulative Error Distribution', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/cumulative_error_distribution.png", dpi=300)
    plt.show()

def plot_component_errors_violin(component_errors_mle, component_errors_recon, plots_dir):
    plt.figure(figsize=(12, 8))
    components = ['X', 'Y', 'Z']
    positions = np.arange(len(components))
    
    violin_parts_mle = plt.violinplot([component_errors_mle[:, i] for i in range(3)], 
                                      positions=positions-0.2, 
                                      widths=0.3, 
                                      showmeans=True)
    for pc in violin_parts_mle['bodies']:
        pc.set_facecolor('blue')
        pc.set_alpha(0.5)
    violin_parts_mle['cmeans'].set_color('blue')
    
    violin_parts_recon = plt.violinplot([component_errors_recon[:, i] for i in range(3)], 
                                       positions=positions+0.2, 
                                       widths=0.3, 
                                       showmeans=True)
    for pc in violin_parts_recon['bodies']:
        pc.set_facecolor('orange')
        pc.set_alpha(0.5)
    violin_parts_recon['cmeans'].set_color('orange')
    
    plt.xticks(positions, components, fontsize=14)
    plt.ylabel('Error (units)', fontsize=14)
    plt.title('Component-wise Error Distribution', fontsize=16)
    plt.grid(alpha=0.3, axis='y')
    plt.legend([violin_parts_mle['bodies'][0], violin_parts_recon['bodies'][0]], 
               ['Raw MLE', 'Reconstruction MLE'], 
               fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/component_errors_violin.png", dpi=300)
    plt.show()

def plot_spatial_error_distribution(all_vertex_positions, true_vs_mle_errors, true_vs_recon_errors, plots_dir):
    fig = plt.figure(figsize=(18, 8))
    
    # MLE Spatial Errors
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(all_vertex_positions[:, 0], 
                     all_vertex_positions[:, 1], 
                     all_vertex_positions[:, 2], 
                     c=true_vs_mle_errors,
                     cmap='viridis', 
                     s=15,
                     alpha=0.7,
                     vmin=0,
                     vmax=max(np.percentile(true_vs_mle_errors, 95), np.percentile(true_vs_recon_errors, 95)))
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_zlabel('Z Position', fontsize=12)
    ax1.set_title('Raw MLE Spatial Error Distribution', fontsize=14)
    
    # Reconstruction Spatial Errors
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(all_vertex_positions[:, 0], 
                     all_vertex_positions[:, 1], 
                     all_vertex_positions[:, 2], 
                     c=true_vs_recon_errors,
                     cmap='viridis', 
                     s=15,
                     alpha=0.7,
                     vmin=0,
                     vmax=max(np.percentile(true_vs_mle_errors, 95), np.percentile(true_vs_recon_errors, 95)))
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.set_zlabel('Z Position', fontsize=12)
    ax2.set_title('Reconstruction MLE Spatial Error Distribution', fontsize=14)
    
    # Add colorbar
    cbar = fig.colorbar(sc2, ax=[ax1, ax2], shrink=0.6, pad=0.05)
    cbar.set_label('Error Magnitude', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/spatial_error_distribution.png", dpi=300)
    plt.show()

def plot_error_vs_distance(all_vertex_positions, true_vs_mle_errors, true_vs_recon_errors, params, plots_dir):
    center = np.array([params.R/2, params.C/2, params.T/2])
    distances_from_center = np.sqrt(np.sum((all_vertex_positions - center)**2, axis=1))
    
    plt.figure(figsize=(10, 6))
    
    # Calculate binned statistics for error vs distance
    nbins = 20
    bin_means_mle, bin_edges, binnumber = stats.binned_statistic(
        distances_from_center, true_vs_mle_errors, statistic='mean', bins=nbins)
    bin_means_recon, _, _ = stats.binned_statistic(
        distances_from_center, true_vs_recon_errors, statistic='mean', bins=nbins)
    bin_std_mle, _, _ = stats.binned_statistic(
        distances_from_center, true_vs_mle_errors, statistic='std', bins=nbins)
    bin_std_recon, _, _ = stats.binned_statistic(
        distances_from_center, true_vs_recon_errors, statistic='std', bins=nbins)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.errorbar(bin_centers, bin_means_mle, yerr=bin_std_mle, 
                 fmt='o-', capsize=4, label='Raw MLE', color='blue', markersize=6)
    plt.errorbar(bin_centers, bin_means_recon, yerr=bin_std_recon, 
                 fmt='o-', capsize=4, label='Reconstruction MLE', color='orange', markersize=6)
    
    plt.xlabel('Distance from Center', fontsize=14)
    plt.ylabel('Error (Mean ± Std)', fontsize=14)
    plt.title('Error vs Distance from Center', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/error_vs_distance.png", dpi=300)
    plt.show()

def plot_error_correlation(true_vs_mle_errors, true_vs_recon_errors, plots_dir):
    plt.figure(figsize=(8, 8))
    plt.scatter(true_vs_mle_errors, true_vs_recon_errors, alpha=0.3, s=20)
    plt.plot([0, max(true_vs_mle_errors.max(), true_vs_recon_errors.max())], 
             [0, max(true_vs_mle_errors.max(), true_vs_recon_errors.max())], 
             'k--', label='y=x')
    
    # Calculate and add correlation
    corr = np.corrcoef(true_vs_mle_errors, true_vs_recon_errors)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Raw MLE Error', fontsize=14)
    plt.ylabel('Reconstruction MLE Error', fontsize=14)
    plt.title('Error Correlation Analysis', fontsize=16)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/error_correlation.png", dpi=300)
    plt.show()

def plot_per_vertex_errors(N, true_vs_mle_errors, true_vs_recon_errors, params, plots_dir):
    plt.figure(figsize=(12, 8))
    
    # Reshape errors to have vertex indices
    vertex_indices = np.repeat(np.arange(N), len(true_vs_mle_errors) // N)
    
    # Calculate per-vertex errors
    vertex_mle_errors = [true_vs_mle_errors[vertex_indices == i] for i in range(N)]
    vertex_recon_errors = [true_vs_recon_errors[vertex_indices == i] for i in range(N)]
    
    positions = np.arange(N)
    width = 0.35
    
    # Create box plots
    bp1 = plt.boxplot(vertex_mle_errors, 
                     positions=positions-width/2, 
                     widths=width, 
                     patch_artist=True,
                     medianprops=dict(color='black'))
    bp2 = plt.boxplot(vertex_recon_errors, 
                     positions=positions+width/2, 
                     widths=width, 
                     patch_artist=True,
                     medianprops=dict(color='black'))
    
    # Set colors
    for box in bp1['boxes']:
        box.set(facecolor='blue', alpha=0.5)
    for box in bp2['boxes']:
        box.set(facecolor='orange', alpha=0.5)
    
    plt.xlabel('Vertex Index', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title('Per-vertex Error Analysis', fontsize=16)
    plt.xticks(positions)
    plt.grid(alpha=0.3, axis='y')
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Raw MLE', 'Reconstruction MLE'], fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/per_vertex_errors.png", dpi=300)
    plt.show()

def plot_radial_angular_errors(all_vertex_positions, all_mle_vertex_positions, all_reconstruction_vertex_positions, plots_dir):
    # Compute radial and angular errors (relative to true positions)
    radial_mle_errors = np.abs(np.sqrt(np.sum(all_mle_vertex_positions**2, axis=1)) - 
                              np.sqrt(np.sum(all_vertex_positions**2, axis=1)))
    radial_recon_errors = np.abs(np.sqrt(np.sum(all_reconstruction_vertex_positions**2, axis=1)) - 
                               np.sqrt(np.sum(all_vertex_positions**2, axis=1)))
    
    # Compute angular errors (dot product of normalized vectors)
    def angular_error(v1, v2):
        # Handle zero vectors safely
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        zero_mask = (norm1 == 0) | (norm2 == 0)
        
        # Initialize with zeros
        angles = np.zeros(len(v1))
        
        # Compute only for non-zero vectors
        non_zero = ~zero_mask
        if np.any(non_zero):
            v1_norm = v1[non_zero] / norm1[non_zero, np.newaxis]
            v2_norm = v2[non_zero] / norm2[non_zero, np.newaxis]
            cos_angles = np.sum(v1_norm * v2_norm, axis=1)
            # Clip to avoid numerical errors
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            angles[non_zero] = np.arccos(cos_angles) * (180 / np.pi)  # in degrees
        
        return angles
    
    angular_mle_errors = angular_error(all_vertex_positions, all_mle_vertex_positions)
    angular_recon_errors = angular_error(all_vertex_positions, all_reconstruction_vertex_positions)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Radial errors
    sns.kdeplot(radial_mle_errors, ax=ax1, label='Raw MLE', fill=True, alpha=0.3)
    sns.kdeplot(radial_recon_errors, ax=ax1, label='Reconstruction MLE', fill=True, alpha=0.3)
    ax1.set_xlabel('Radial Error', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.set_title('Radial Error Distribution', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Angular errors
    sns.kdeplot(angular_mle_errors, ax=ax2, label='Raw MLE', fill=True, alpha=0.3)
    sns.kdeplot(angular_recon_errors, ax=ax2, label='Reconstruction MLE', fill=True, alpha=0.3)
    ax2.set_xlabel('Angular Error (degrees)', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_title('Angular Error Distribution', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/radial_angular_errors.png", dpi=300)
    plt.show()

def plot_bootstrap_analysis(true_vs_mle_errors, true_vs_recon_errors, mle_mean, recon_mean, plots_dir):
    plt.figure(figsize=(10, 6))
    
    n_bootstrap = 1000
    bootstrap_differences = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(true_vs_mle_errors), len(true_vs_mle_errors), replace=True)
        mle_sample = true_vs_mle_errors[indices]
        recon_sample = true_vs_recon_errors[indices]
        bootstrap_differences.append(np.mean(mle_sample) - np.mean(recon_sample))
    
    bootstrap_differences = np.array(bootstrap_differences)
    
    # Plot bootstrap distribution
    sns.histplot(bootstrap_differences, kde=True)
    
    # Add confidence interval lines
    ci_low = np.percentile(bootstrap_differences, 2.5)
    ci_high = np.percentile(bootstrap_differences, 97.5)
    plt.axvline(ci_low, color='red', linestyle='--', label=f'95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
    plt.axvline(ci_high, color='red', linestyle='--')
    
    # Add observed difference
    observed_diff = mle_mean - recon_mean
    plt.axvline(observed_diff, color='green', linewidth=2, 
               label=f'Observed: {observed_diff:.4f}')
    
    plt.xlabel('Raw MLE Error - Reconstruction MLE Error', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Bootstrap Analysis of Error Difference', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/bootstrap_significance.png", dpi=300)
    plt.show()
    
    return ci_low, ci_high, observed_diff

def create_summary_report(model_name, mle_mean, recon_mean, improvement_pct, mle_median, recon_median, 
                         mle_std, recon_std, mle_95pct, recon_95pct, component_errors_mle, 
                         component_errors_recon, plots_dir):
    # Calculate bootstrap confidence interval
    ci_low, ci_high, observed_diff = 0, 0, mle_mean - recon_mean  # Placeholder values
    corr = np.corrcoef(np.mean(np.abs(component_errors_mle), axis=0), 
                       np.mean(np.abs(component_errors_recon), axis=0))[0, 1]
    
    with open(f"{plots_dir}/analysis_summary.txt", 'w') as f:
        f.write(f"Likelihood Analysis Summary for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Basic Statistics:\n")
        f.write(f"MLE Mean Error: {mle_mean:.4f}\n")
        f.write(f"Reconstruction Mean Error: {recon_mean:.4f}\n")
        f.write(f"Improvement: {improvement_pct:.2f}%\n\n")
        
        f.write(f"MLE Median Error: {mle_median:.4f}\n")
        f.write(f"Reconstruction Median Error: {recon_median:.4f}\n\n")
        
        f.write(f"MLE Error Std Dev: {mle_std:.4f}\n")
        f.write(f"Reconstruction Error Std Dev: {recon_std:.4f}\n\n")
        
        f.write(f"MLE 95th Percentile Error: {mle_95pct:.4f}\n")
        f.write(f"Reconstruction 95th Percentile Error: {recon_95pct:.4f}\n\n")
        
        f.write("Bootstrap Analysis:\n")
        f.write(f"95% Confidence Interval for Error Difference: [{ci_low:.4f}, {ci_high:.4f}]\n")
        f.write(f"Observed Difference: {observed_diff:.4f}\n")
        f.write(f"Statistically Significant: {ci_low > 0}\n\n")
        
        f.write("Error Correlation:\n")
        f.write(f"Correlation between MLE and Reconstruction Errors: {corr:.4f}\n\n")
        
        f.write("Component-wise Analysis:\n")
        for i, comp in enumerate(['X', 'Y', 'Z']):
            f.write(f"{comp} Component - MLE Mean Error: {np.mean(np.abs(component_errors_mle[:, i])):.4f}\n")
            f.write(f"{comp} Component - Reconstruction Mean Error: {np.mean(np.abs(component_errors_recon[:, i])):.4f}\n")


def compute_pulse_metrics(waveform, time_axis):
    """
    Compute pulse metrics for a waveform
    
    Parameters:
    -----------
    waveform : ndarray
        Pulse waveform data
    time_axis : ndarray
        Time axis for the waveform
        
    Returns:
    --------
    dict
        Dictionary of pulse metrics
    """
    # Find pulse peak
    peak_idx = np.argmax(waveform)
    peak_amplitude = waveform[peak_idx]
    peak_time = time_axis[peak_idx]
    
    # Calculate pulse area (integral)
    pulse_area = np.sum(waveform)
    
    # Calculate 10/90 rise time
    threshold_10 = 0.1 * peak_amplitude
    threshold_90 = 0.9 * peak_amplitude
    
    # Find index where pulse crosses 10% threshold (rising edge)
    rise_10_idx = np.where(waveform[:peak_idx] >= threshold_10)[0]
    rise_10_time = time_axis[rise_10_idx[0]] if len(rise_10_idx) > 0 else time_axis[0]
    
    # Find index where pulse crosses 90% threshold (rising edge)
    rise_90_idx = np.where(waveform[:peak_idx] >= threshold_90)[0]
    rise_90_time = time_axis[rise_90_idx[0]] if len(rise_90_idx) > 0 else peak_time
    
    rise_time = rise_90_time - rise_10_time
    
    # Calculate 10/90 fall time
    fall_90_idx = np.where(waveform[peak_idx:] <= threshold_90)[0]
    fall_90_time = time_axis[peak_idx + fall_90_idx[0]] if len(fall_90_idx) > 0 else peak_time
    
    fall_10_idx = np.where(waveform[peak_idx:] <= threshold_10)[0]
    fall_10_time = time_axis[peak_idx + fall_10_idx[0]] if len(fall_10_idx) > 0 else time_axis[-1]
    
    fall_time = fall_10_time - fall_90_time
    
    # Calculate FWHM (Full Width at Half Maximum)
    half_max = 0.5 * peak_amplitude
    left_idx = np.where(waveform[:peak_idx] >= half_max)[0]
    left_time = time_axis[left_idx[0]] if len(left_idx) > 0 else time_axis[0]
    
    right_idx = np.where(waveform[peak_idx:] <= half_max)[0]
    right_time = time_axis[peak_idx + right_idx[0]] if len(right_idx) > 0 else time_axis[-1]
    
    fwhm = right_time - left_time
    
    # Calculate prompt fraction (first 100 ns after 10% rise)
    prompt_window = 100  # ns
    prompt_end_time = rise_10_time + prompt_window
    prompt_end_idx = np.searchsorted(time_axis, prompt_end_time)
    prompt_area = np.trapz(waveform[:prompt_end_idx], time_axis[:prompt_end_idx])
    prompt_fraction = prompt_area / pulse_area if pulse_area > 0 else 0
    
    # Calculate 10/90 width (distance between 10% and 90% of cumulative area)
    cumulative_area = np.cumsum(waveform)
    total_area = cumulative_area[-1]
    if total_area > 0:
        area_10_idx = np.searchsorted(cumulative_area, 0.1 * total_area)
        area_90_idx = np.searchsorted(cumulative_area, 0.9 * total_area)
        width_1090 = time_axis[area_90_idx] - time_axis[area_10_idx]
    else:
        width_1090 = 0
    
    return {
        'peak_amplitude': peak_amplitude,
        'peak_time': peak_time,
        'pulse_area': pulse_area,
        'rise_time': rise_time,
        'fall_time': fall_time,
        'fwhm': fwhm,
        'prompt_fraction': prompt_fraction,
        'width_1090': width_1090
    }

def compute_waveform_metrics(waveform, time_axis):
    """Compute pulse metrics for a single waveform"""
    # Skip empty channels
    if np.max(waveform) > 0:
        return compute_pulse_metrics(waveform, time_axis)
    return None

def rq_analysis(model, samples=2048, tune=False, loss=None):
    params = Params()
    batch_size = 32
    data_generator = vertex_electron_batch_generator([1, 2, 3, 4], params, batch_size)
    validation_data_generator = vertex_electron_batch_generator([1, 2, 3, 4], params, batch_size)

    if tune:
        def autoencoder_generator_from_generator(generator):
            while True:
                x, y, N = next(iter(generator))
                x = np.array(np.reshape(x, (batch_size, params.R * params.C, params.T)), dtype=np.float32)
                yield x, x

        if loss is None:
            print('ERROR: if tuning, please provide a loss')
            return
        
        autoencoder_data_generator = autoencoder_generator_from_generator(data_generator)
        validation_autoencoder_data_generator = autoencoder_generator_from_generator(validation_data_generator)
        train_models(
            [model], [loss], [tf.keras.optimizers.Adam(3e-4)], 
            autoencoder_data_generator, validation_autoencoder_data_generator, 
            epochs=0, steps_per_epoch=64, batch_size=batch_size, 
            use_checkpoints=True,
            ckpt_dir=f'ckpts/likelihood_tuning_'
        )

    plots_dir = f"rq_analysis_{model.name}"
    os.makedirs(plots_dir, exist_ok=True)
    plot_rq_visualization(model, data_generator, plots_dir)

    raw_metrics_list = []
    recon_metrics_list = []
    
    time_axis = np.arange(params.T) * SAMPLERATE
    
    batch_size = 64
    data_generator = vertex_electron_batch_generator([1], params, batch_size)
    for batch in range(0, samples, batch_size):
        current_batch_size = min(batch_size, samples - batch)
        raw_data, _, _ = next(iter(data_generator))
        raw_data = raw_data[:current_batch_size]
        
        raw_data_ = np.array(np.reshape(raw_data, (current_batch_size, params.R * params.C, params.T)), dtype=np.float32)
        
        try:
            autoencoder_reconstruction_, _, _, _ = model(raw_data_)
        except:
            autoencoder_reconstruction_ = model(raw_data_)
        autoencoder_reconstruction = np.reshape(autoencoder_reconstruction_, (current_batch_size, params.R, params.C, params.T))
        
        raw_data_reshaped = raw_data.reshape(current_batch_size, params.R * params.C, params.T)
        recon_data_reshaped = autoencoder_reconstruction.reshape(current_batch_size, params.R * params.C, params.T)
        for i in tqdm(range(current_batch_size), desc='computing pulse metrics', ncols=120):
            raw_summed_waveform = np.sum(raw_data_reshaped[i], axis=0)  # Sum over all R*C channels
            recon_summed_waveform = np.sum(recon_data_reshaped[i], axis=0)  # Sum over all R*C channels
            
            raw_metrics = compute_waveform_metrics(raw_summed_waveform, time_axis)
            if raw_metrics:
                raw_metrics_list.append(raw_metrics)
            
            recon_metrics = compute_waveform_metrics(recon_summed_waveform, time_axis)
            if recon_metrics:
                recon_metrics_list.append(recon_metrics)
    
    # Convert lists of metrics to structured arrays for easier analysis
    raw_metrics_array = {
        'peak_amplitude': np.array([m['peak_amplitude'] for m in raw_metrics_list if m]),
        'pulse_area': np.array([m['pulse_area'] for m in raw_metrics_list if m]),
        'rise_time': np.array([m['rise_time'] for m in raw_metrics_list if m]),
        'fall_time': np.array([m['fall_time'] for m in raw_metrics_list if m]),
        'fwhm': np.array([m['fwhm'] for m in raw_metrics_list if m]),
        'prompt_fraction': np.array([m['prompt_fraction'] for m in raw_metrics_list if m]),
        'width_1090': np.array([m['width_1090'] for m in raw_metrics_list if m])
    }
    
    recon_metrics_array = {
        'peak_amplitude': np.array([m['peak_amplitude'] for m in recon_metrics_list if m]),
        'pulse_area': np.array([m['pulse_area'] for m in recon_metrics_list if m]),
        'rise_time': np.array([m['rise_time'] for m in recon_metrics_list if m]),
        'fall_time': np.array([m['fall_time'] for m in recon_metrics_list if m]),
        'fwhm': np.array([m['fwhm'] for m in recon_metrics_list if m]),
        'prompt_fraction': np.array([m['prompt_fraction'] for m in recon_metrics_list if m]),
        'width_1090': np.array([m['width_1090'] for m in recon_metrics_list if m])
    }
    
    plot_rq_distributions(raw_metrics_array, recon_metrics_array, plots_dir)
    plot_rq_correlation(raw_metrics_array, recon_metrics_array, plots_dir)
    plot_rq_fractional_errors(raw_metrics_array, recon_metrics_array, plots_dir)
    plot_rq_metric_relative_difference_violin(raw_metrics_array, recon_metrics_array, plots_dir)
    plot_rq_reconstruction_quality_radar(raw_metrics_array, recon_metrics_array, plots_dir)
    # plot_rq_3d_metric_distribution(raw_metrics_array, recon_metrics_array, plots_dir)
    # plot_rq_metric_time_evolution(raw_metrics_array, recon_metrics_array, plots_dir)
    
    # Only generate waveform examples if we have access to the data generator
    if 'validation_data_generator' in locals() or 'data_generator' in locals():
        try:
            generator = locals().get('validation_data_generator', locals().get('data_generator'))
            plot_rq_waveform_examples(model, generator, plots_dir, num_examples=4)
        except Exception as e:
            print(f"Could not generate waveform examples: {e}")
    
    plot_rq_summary_table(raw_metrics_array, recon_metrics_array, plots_dir)
    
    # Calculate summary statistics for a report
    report = create_rq_summary_report(model.name, raw_metrics_array, recon_metrics_array, plots_dir)
    
    print(f"\nReconstruction Quality Analysis complete. Visualizations and report saved to: {plots_dir}/")
    
    return {
        'raw_metrics': raw_metrics_array,
        'recon_metrics': recon_metrics_array,
        'plots_dir': plots_dir,
        'report': report
    }

def plot_rq_visualization(model, data_generator, plots_dir):
    params = Params()
    test_events, _, _ = next(iter(data_generator))
    test_events = test_events.astype(np.float32).reshape((test_events.shape[0], params.R * params.C, params.T))
    test_event = test_events[0].reshape((params.R, params.C, params.T))
    convolved_test_event = gaussian_blur_3d(test_event[np.newaxis, :, :, :, np.newaxis], kernel_size=5, sigma=1)
    try:
        autoencoder_reconstruction = model(test_events)[0][0].numpy().reshape((params.R, params.C, params.T))
    except:
        autoencoder_reconstruction = model(test_events)[0].numpy().reshape((params.R, params.C, params.T))
    
    display_events = [test_event, convolved_test_event, autoencoder_reconstruction]
    display_events_titles = [
        'Original', 
        'Original (Gaussian blur, σ=1)', 
        f'GVQ-VAE Reconstruction\nReconstruction loss (MSE): {reconstruction_loss(test_event, autoencoder_reconstruction).numpy() : .3f} phd^2', 
    ]
    
    plot_events(np.stack(display_events, axis=0), title='', subtitles=display_events_titles)

def plot_rq_distributions(raw_metrics, recon_metrics, plots_dir):
    """
    Plot distributions of various reconstruction quality metrics.
    
    Parameters:
    -----------
    raw_metrics : dict
        Dictionary of raw waveform metrics
    recon_metrics : dict
        Dictionary of reconstructed waveform metrics
    plots_dir : str
        Directory to save plots
    """
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    titles = ['Pulse Area (phd)', '10-90% Rise Time (ns)', '90-10% Fall Time (ns)', 
              'FWHM (ns)', 'Peak Amplitude (phd-ns)', 'Prompt Fraction', '10-90% Width (ns)']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Get raw and reconstructed metrics
        raw_values = raw_metrics[metric]
        recon_values = recon_metrics[metric]
        
        # Remove outliers for better visualization (cap at 99th percentile)
        raw_99p = np.percentile(raw_values, 99)
        recon_99p = np.percentile(recon_values, 99)
        max_val = max(raw_99p, recon_99p)
        
        raw_values_clipped = raw_values[raw_values <= max_val]
        recon_values_clipped = recon_values[recon_values <= max_val]
        
        # Calculate statistics
        raw_mean = np.mean(raw_values)
        recon_mean = np.mean(recon_values)
        raw_median = np.median(raw_values)
        recon_median = np.median(recon_values)
        
        # Plot distributions
        sns.histplot(raw_values_clipped, ax=ax, stat='density', kde=True, 
                    color='blue', alpha=0.4, label='Raw')
        sns.histplot(recon_values_clipped, ax=ax, stat='density', kde=True, 
                    color='orange', alpha=0.4, label='Reconstructed')
        
        # Add vertical lines for means
        ax.axvline(raw_mean, color='blue', linestyle='--', 
                  label=f'Raw Mean: {raw_mean:.2f}')
        ax.axvline(recon_mean, color='orange', linestyle='--', 
                  label=f'Recon Mean: {recon_mean:.2f}')
        
        # Calculate improvement percentage
        if metric == 'rise_time' or metric == 'fall_time' or metric == 'fwhm':
            # For timing metrics, smaller is better
            improvement = (raw_mean - recon_mean) / raw_mean * 100
            if improvement > 0:
                improvement_text = f"Improvement: {improvement:.1f}%"
            else:
                improvement_text = f"Degradation: {-improvement:.1f}%"
        else:
            # For amplitude and area metrics, preservation is key
            rel_diff = (recon_mean - raw_mean) / raw_mean * 100
            if abs(rel_diff) < 5:
                improvement_text = f"Well Preserved: {abs(rel_diff):.1f}% diff"
            else:
                improvement_text = f"Difference: {rel_diff:.1f}%"
        
        # Add improvement text
        ax.text(0.5, 0.95, improvement_text, transform=ax.transAxes, 
               horizontalalignment='center', verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_distributions.png", dpi=300)
    plt.close(fig)

def plot_rq_correlation(raw_metrics, recon_metrics, plots_dir):
    """Plot correlations between raw and reconstructed metrics"""
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    titles = ['Pulse Area (phd)', '10-90% Rise Time (ns)', '90-10% Fall Time (ns)', 
              'FWHM (ns)', 'Peak Amplitude (phd-ns)', 'Prompt Fraction', '10-90% Width (ns)']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Get raw and reconstructed metrics
        raw_values = raw_metrics[metric]
        recon_values = recon_metrics[metric]
        
        # Match lengths if necessary
        n = min(len(raw_values), len(recon_values))
        raw_values = raw_values[:n]
        recon_values = recon_values[:n]
        
        # Remove outliers for better visualization (cap at 99th percentile)
        raw_99p = np.percentile(raw_values, 99)
        recon_99p = np.percentile(recon_values, 99)
        max_val = max(raw_99p, recon_99p)
        
        # Create mask for valid points
        valid_mask = (raw_values <= max_val) & (recon_values <= max_val)
        raw_values_valid = raw_values[valid_mask]
        recon_values_valid = recon_values[valid_mask]
        
        # Calculate correlation
        corr = np.corrcoef(raw_values_valid, recon_values_valid)[0, 1]
        
        # Create scatter plot with hexbin for dense regions
        hb = ax.hexbin(raw_values_valid, recon_values_valid, 
                      gridsize=50, cmap='viridis', 
                      mincnt=1, bins='log')
        
        # Add correlation line
        max_range = max(np.max(raw_values_valid), np.max(recon_values_valid))
        min_range = min(np.min(raw_values_valid), np.min(recon_values_valid))
        ax.plot([min_range, max_range], [min_range, max_range], 'r--', 
               label='y=x')
        
        # Add correlation coefficient
        ax.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.set_xlabel(f'Raw {title}')
        ax.set_ylabel(f'Reconstructed {title}')
        ax.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('Count (log scale)')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_correlation.png", dpi=300)
    plt.close(fig)

def plot_rq_fractional_errors(raw_metrics, recon_metrics, plots_dir):
    """Plot distributions of fractional errors for each metric"""
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    titles = ['Pulse Area (phd)', '10-90% Rise Time (ns)', '90-10% Fall Time (ns)', 
              'FWHM (ns)', 'Peak Amplitude (phd-ns)', 'Prompt Fraction', '10-90% Width (ns)']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        # Get raw and reconstructed metrics
        raw_values = raw_metrics[metric]
        recon_values = recon_metrics[metric]
        
        # Match lengths if necessary
        n = min(len(raw_values), len(recon_values))
        raw_values = raw_values[:n]
        recon_values = recon_values[:n]
        
        # Create mask for valid points (non-zero raw values)
        valid_mask = (raw_values > 0)
        raw_values_valid = raw_values[valid_mask]
        recon_values_valid = recon_values[valid_mask]
        
        # Calculate fractional errors
        fractional_errors = (recon_values_valid - raw_values_valid) / raw_values_valid
        
        # Remove extreme outliers (beyond ±100%)
        valid_errors = fractional_errors[np.abs(fractional_errors) <= 1.0]
        
        # Plot distribution
        sns.histplot(valid_errors, ax=ax, stat='density', kde=True, 
                    color='green', alpha=0.6)
        
        # Calculate statistics
        mean_error = np.mean(valid_errors)
        median_error = np.median(valid_errors)
        std_error = np.std(valid_errors)
        
        # Add vertical lines for mean and median
        ax.axvline(mean_error, color='red', linestyle='--', 
                  label=f'Mean: {mean_error:.3f}')
        ax.axvline(median_error, color='black', linestyle='-', 
                  label=f'Median: {median_error:.3f}')
        
        # Add text for standard deviation
        ax.text(0.05, 0.95, f'σ = {std_error:.3f}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{title} - Fractional Error')
        ax.set_xlabel('(Reconstructed - Raw) / Raw')
        ax.set_ylabel('Density')
        ax.set_xlim(-1, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_fractional_errors.png", dpi=300)
    plt.close(fig)

def plot_rq_scatter_matrix(raw_metrics, recon_metrics, plots_dir):
    """Create a scatter matrix to visualize relationships between different metrics"""
    # Prepare data for scatter matrix
    raw_df = pd.DataFrame({
        'Pulse Area': raw_metrics['pulse_area'],
        'Rise Time': raw_metrics['rise_time'],
        'Fall Time': raw_metrics['fall_time'],
        'FWHM': raw_metrics['fwhm'],
        'Type': 'Raw'
    })
    
    recon_df = pd.DataFrame({
        'Pulse Area': recon_metrics['pulse_area'],
        'Rise Time': recon_metrics['rise_time'],
        'Fall Time': recon_metrics['fall_time'],
        'FWHM': recon_metrics['fwhm'],
        'Type': 'Reconstructed'
    })
    
    # Combine dataframes
    combined_df = pd.concat([raw_df, recon_df])
    
    # Create scatter matrix
    sns.set(style="ticks")
    scatter_matrix = sns.pairplot(combined_df, 
                                 vars=['Pulse Area', 'Rise Time', 'Fall Time', 'FWHM'],
                                 hue='Type', 
                                 palette={'Raw': 'blue', 'Reconstructed': 'orange'},
                                 diag_kind='kde',
                                 plot_kws={'s': 15, 'alpha': 0.5},
                                 height=3)
    
    scatter_matrix.fig.suptitle('Relationships Between Pulse Metrics', 
                               size=16, y=1.02)
    
    plt.savefig(f"{plots_dir}/rq_scatter_matrix.png", dpi=300)
    plt.close(scatter_matrix.fig)

def plot_rq_position_dependence(raw_metrics, recon_metrics, plots_dir):
    """Placeholder for position-dependent analysis - would require position information"""
    # This is a simplified version since we don't have position data in this example
    # In a real implementation, you would use vertex positions from the generator
    
    # Create a figure showing how metrics vary with detector position
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.text(0.5, 0.5, "Position-dependent analysis requires vertex position data\nwhich would be included in a complete implementation",
           horizontalalignment='center', verticalalignment='center',
           transform=ax.transAxes, fontsize=14)
    
    ax.set_title("Pulse Metrics vs. Position (Placeholder)")
    ax.axis('off')
    
    plt.savefig(f"{plots_dir}/rq_position_dependence.png", dpi=300)
    plt.close(fig)

def plot_rq_summary_table(raw_metrics, recon_metrics, plots_dir):
    """Create a summary table of all pulse metrics"""
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    titles = ['Pulse Area (phd)', '10-90% Rise Time (ns)', '90-10% Fall Time (ns)', 
              'FWHM (ns)', 'Peak Amplitude (phd-ns)', 'Prompt Fraction', '10-90% Width (ns)']
    
    # Calculate summary statistics
    summary_data = []
    
    for metric, title in zip(metrics, titles):
        raw_values = raw_metrics[metric]
        recon_values = recon_metrics[metric]
        
        # Match lengths if necessary
        n = min(len(raw_values), len(recon_values))
        raw_values = raw_values[:n]
        recon_values = recon_values[:n]
        
        # Create mask for valid points (non-zero raw values)
        valid_mask = (raw_values > 0)
        raw_values_valid = raw_values[valid_mask]
        recon_values_valid = recon_values[valid_mask]
        
        # Calculate statistics
        raw_mean = np.mean(raw_values_valid)
        raw_median = np.median(raw_values_valid)
        raw_std = np.std(raw_values_valid)
        
        recon_mean = np.mean(recon_values_valid)
        recon_median = np.median(recon_values_valid)
        recon_std = np.std(recon_values_valid)
        
        # Calculate fractional errors
        fractional_errors = (recon_values_valid - raw_values_valid) / raw_values_valid
        mean_frac_error = np.mean(fractional_errors)
        median_frac_error = np.median(fractional_errors)
        std_frac_error = np.std(fractional_errors)
        
        # Calculate correlation
        corr = np.corrcoef(raw_values_valid, recon_values_valid)[0, 1]
        
        # Add to summary data
        summary_data.append([
            title,
            f"{raw_mean:.3f} ± {raw_std:.3f}",
            f"{recon_mean:.3f} ± {recon_std:.3f}",
            f"{mean_frac_error*100:.2f}%",
            f"{corr:.4f}"
        ])
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    table = ax.table(
        cellText=summary_data,
        colLabels=['Metric', 'Raw (Mean ± Std)', 'Reconstructed (Mean ± Std)', 
                   'Mean Frac. Error', 'Correlation'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Pulse Metrics Summary Table', fontsize=16, pad=20)
    plt.tight_layout()
    
    plt.savefig(f"{plots_dir}/rq_summary_table.png", dpi=300)
    plt.close(fig)
    
    # Also save as CSV for reference
    with open(f"{plots_dir}/rq_summary_table.csv", 'w') as f:
        f.write('Metric,Raw (Mean ± Std),Reconstructed (Mean ± Std),Mean Frac. Error,Correlation\n')
        for row in summary_data:
            f.write(','.join(row) + '\n')

def create_rq_summary_report(model_name, raw_metrics, recon_metrics, plots_dir):
    """Create a detailed summary report of reconstruction quality metrics"""
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    metric_names = ['Pulse Area', 'Rise Time (10-90%)', 'Fall Time (90-10%)', 
                   'FWHM', 'Peak Amplitude', 'Prompt Fraction', '10-90% Width (ns)']
    
    report_path = f"{plots_dir}/rq_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"Reconstruction Quality Analysis Report for {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Summary of Pulse Shape Reconstruction Quality\n")
        f.write("-" * 60 + "\n")
        
        for metric, name in zip(metrics, metric_names):
            raw_values = raw_metrics[metric]
            recon_values = recon_metrics[metric]
            
            # Match lengths if necessary
            n = min(len(raw_values), len(recon_values))
            raw_values = raw_values[:n]
            recon_values = recon_values[:n]
            
            # Create mask for valid points (non-zero raw values)
            valid_mask = (raw_values > 0)
            raw_values_valid = raw_values[valid_mask]
            recon_values_valid = recon_values[valid_mask]
            
            # Basic statistics
            raw_mean = np.mean(raw_values_valid)
            raw_median = np.median(raw_values_valid)
            raw_std = np.std(raw_values_valid)
            
            recon_mean = np.mean(recon_values_valid)
            recon_median = np.median(recon_values_valid)
            recon_std = np.std(recon_values_valid)
            
            # Fractional errors
            fractional_errors = (recon_values_valid - raw_values_valid) / raw_values_valid
            mean_frac_error = np.mean(fractional_errors)
            median_frac_error = np.median(fractional_errors)
            std_frac_error = np.std(fractional_errors)
            
            # 95% confidence interval for fractional error
            ci_low = np.percentile(fractional_errors, 2.5)
            ci_high = np.percentile(fractional_errors, 97.5)
            
            # Correlation
            corr = np.corrcoef(raw_values_valid, recon_values_valid)[0, 1]
            
            # Write metric details
            f.write(f"\n{name}:\n")
            f.write(f"  Raw Data:           Mean = {raw_mean:.4f}, Median = {raw_median:.4f}, Std = {raw_std:.4f}\n")
            f.write(f"  Reconstructed Data: Mean = {recon_mean:.4f}, Median = {recon_median:.4f}, Std = {recon_std:.4f}\n")
            f.write(f"  Mean Fractional Error: {mean_frac_error*100:.2f}%\n")
            f.write(f"  Median Fractional Error: {median_frac_error*100:.2f}%\n")
            f.write(f"  95% CI for Fractional Error: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]\n")
            f.write(f"  Correlation Coefficient: {corr:.4f}\n")
            
            # Assess reconstruction quality
            quality_score = abs(corr) * (1 - min(abs(mean_frac_error), 1.0))
            
            if quality_score > 0.9:
                quality = "Excellent"
            elif quality_score > 0.8:
                quality = "Very Good"
            elif quality_score > 0.7:
                quality = "Good"
            elif quality_score > 0.5:
                quality = "Fair"
            else:
                quality = "Poor"
                
            f.write(f"  Reconstruction Quality: {quality} (Score: {quality_score:.4f})\n")
        
        # Overall assessment
        f.write("\nOverall Reconstruction Assessment:\n")
        f.write("-" * 60 + "\n")
        
        # Calculate overall correlation across all metrics
        overall_correlations = []
        for metric in metrics:
            raw_values = raw_metrics[metric]
            recon_values = recon_metrics[metric]
            
            # Match lengths if necessary
            n = min(len(raw_values), len(recon_values))
            raw_values = raw_values[:n]
            recon_values = recon_values[:n]
            
            # Create mask for valid points (non-zero raw values)
            valid_mask = (raw_values > 0)
            raw_values_valid = raw_values[valid_mask]
            recon_values_valid = recon_values[valid_mask]
            
            if len(raw_values_valid) > 0:
                corr = np.corrcoef(raw_values_valid, recon_values_valid)[0, 1]
                overall_correlations.append(corr)
        
        avg_correlation = np.mean(overall_correlations)
        
        f.write(f"Average Correlation Across All Metrics: {avg_correlation:.4f}\n")
        
        if avg_correlation > 0.9:
            recommendation = "The reconstructed pulses show excellent fidelity to the original data."
        elif avg_correlation > 0.8:
            recommendation = "The reconstructed pulses show very good fidelity, suitable for most analysis tasks."
        elif avg_correlation > 0.7:
            recommendation = "The reconstructed pulses show good fidelity, but may introduce some bias in detailed analysis."
        else:
            recommendation = "The reconstructed pulses show significant differences from the original data, caution is advised."
            
        f.write(f"\nConclusion: {recommendation}\n")
    
    print(f"Report saved to {report_path}")
    return report_path

def plot_rq_metric_comparison(raw_metric, recon_metric, metric_name, plots_dir):
    """
    Create individual comparison plots for a single metric.
    
    Parameters:
    -----------
    raw_metric : array
        Raw metric values
    recon_metric : array
        Reconstructed metric values
    metric_name : str
        Name of the metric for plot labels
    plots_dir : str
        Directory to save plot
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Remove outliers for better visualization (cap at 99th percentile)
    raw_99p = np.percentile(raw_metric, 99)
    recon_99p = np.percentile(recon_metric, 99)
    max_val = max(raw_99p, recon_99p)
    
    raw_values_clipped = raw_metric[raw_metric <= max_val]
    recon_values_clipped = recon_metric[recon_metric <= max_val]
    
    # Calculate statistics
    raw_mean = np.mean(raw_metric)
    recon_mean = np.mean(recon_metric)
    raw_median = np.median(raw_metric)
    recon_median = np.median(recon_metric)
    
    # Left plot: Distribution comparison
    sns.histplot(raw_values_clipped, ax=ax1, stat='density', kde=True, 
                color='blue', alpha=0.4, label='Raw')
    sns.histplot(recon_values_clipped, ax=ax1, stat='density', kde=True, 
                color='orange', alpha=0.4, label='Reconstructed')
    
    # Add vertical lines for means
    ax1.axvline(raw_mean, color='blue', linestyle='--', 
               label=f'Raw Mean: {raw_mean:.2f}')
    ax1.axvline(recon_mean, color='orange', linestyle='--', 
               label=f'Recon Mean: {recon_mean:.2f}')
    
    ax1.set_title(f'{metric_name} Distribution')
    ax1.set_xlabel(metric_name)
    ax1.set_ylabel('Density')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Right plot: Correlation scatter
    # Create mask for valid points
    valid_mask = (raw_metric <= max_val) & (recon_metric <= max_val)
    raw_values_valid = raw_metric[valid_mask]
    recon_values_valid = recon_metric[valid_mask]
    
    # Calculate correlation
    corr = np.corrcoef(raw_values_valid, recon_values_valid)[0, 1]
    
    # Create scatter plot with hexbin for dense regions
    hb = ax2.hexbin(raw_values_valid, recon_values_valid, 
                    gridsize=50, cmap='viridis', 
                    mincnt=1, bins='log')
    
    # Add correlation line
    max_range = max(np.max(raw_values_valid), np.max(recon_values_valid))
    min_range = min(np.min(raw_values_valid), np.min(recon_values_valid))
    ax2.plot([min_range, max_range], [min_range, max_range], 'r--', 
              label='y=x')
    
    # Add correlation coefficient
    ax2.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax2.transAxes,
              fontsize=12, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title(f'{metric_name} Correlation')
    ax2.set_xlabel(f'Raw {metric_name}')
    ax2.set_ylabel(f'Reconstructed {metric_name}')
    ax2.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(hb, ax=ax2)
    cbar.set_label('Count (log scale)')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_{metric_name.lower().replace(' ', '_')}_comparison.png", dpi=300)
    plt.close(fig)
    
    return corr

def plot_rq_metric_fractional_error(raw_metric, recon_metric, metric_name, plots_dir):
    """
    Create fractional error plot for a single metric.
    
    Parameters:
    -----------
    raw_metric : array
        Raw metric values
    recon_metric : array
        Reconstructed metric values
    metric_name : str
        Name of the metric for plot labels
    plots_dir : str
        Directory to save plot
    """
    # Create mask for valid points (non-zero raw values)
    valid_mask = (raw_metric > 0)
    raw_values_valid = raw_metric[valid_mask]
    recon_values_valid = recon_metric[valid_mask]
    
    # Calculate fractional errors
    fractional_errors = (recon_values_valid - raw_values_valid) / raw_values_valid
    
    # Remove extreme outliers (beyond ±100%)
    valid_errors = fractional_errors[np.abs(fractional_errors) <= 1.0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot distribution with KDE
    sns.histplot(valid_errors, ax=ax, stat='density', kde=True, 
                color='green', alpha=0.6, bins=30)
    
    # Calculate statistics
    mean_error = np.mean(valid_errors)
    median_error = np.median(valid_errors)
    std_error = np.std(valid_errors)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_error, color='red', linestyle='--', 
               label=f'Mean: {mean_error:.3f}')
    ax.axvline(median_error, color='black', linestyle='-', 
               label=f'Median: {median_error:.3f}')
    
    # Add text for standard deviation
    ax.text(0.05, 0.95, f'σ = {std_error:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'{metric_name} - Fractional Error')
    ax.set_xlabel('(Reconstructed - Raw) / Raw')
    ax.set_ylabel('Density')
    ax.set_xlim(-1, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_{metric_name.lower().replace(' ', '_')}_frac_error.png", dpi=300)
    plt.close(fig)
    
    return mean_error, median_error, std_error

def plot_rq_metric_qq(raw_metric, recon_metric, metric_name, plots_dir):
    """
    Create Q-Q plot for comparing the distributions of raw and reconstructed metric values.
    
    Parameters:
    -----------
    raw_metric : array
        Raw metric values
    recon_metric : array
        Reconstructed metric values
    metric_name : str
        Name of the metric for plot labels
    plots_dir : str
        Directory to save plot
    """
    # Remove outliers for better visualization (cap at 99th percentile)
    raw_99p = np.percentile(raw_metric, 99)
    recon_99p = np.percentile(recon_metric, 99)
    max_val = max(raw_99p, recon_99p)
    
    valid_mask = (raw_metric <= max_val) & (recon_metric <= max_val)
    raw_values_valid = raw_metric[valid_mask]
    recon_values_valid = recon_metric[valid_mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Generate Q-Q plot
    quantiles = np.linspace(0, 1, 100)
    raw_quantiles = np.quantile(raw_values_valid, quantiles)
    recon_quantiles = np.quantile(recon_values_valid, quantiles)
    
    # Plot Q-Q points
    ax.scatter(raw_quantiles, recon_quantiles, alpha=0.7, s=30)
    
    # Add reference line
    max_range = max(np.max(raw_quantiles), np.max(recon_quantiles))
    min_range = min(np.min(raw_quantiles), np.min(recon_quantiles))
    ax.plot([min_range, max_range], [min_range, max_range], 'r--', label='y=x')
    
    ax.set_title(f'Q-Q Plot for {metric_name}')
    ax.set_xlabel(f'Raw {metric_name} Quantiles')
    ax.set_ylabel(f'Reconstructed {metric_name} Quantiles')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_{metric_name.lower().replace(' ', '_')}_qq_plot.png", dpi=300)
    plt.close(fig)

def plot_rq_metric_bland_altman(raw_metric, recon_metric, metric_name, plots_dir):
    """
    Create Bland-Altman plot (difference vs average) for a single metric.
    
    Parameters:
    -----------
    raw_metric : array
        Raw metric values
    recon_metric : array
        Reconstructed metric values
    metric_name : str
        Name of the metric for plot labels
    plots_dir : str
        Directory to save plot
    """
    # Remove outliers for better visualization (cap at 99th percentile)
    raw_99p = np.percentile(raw_metric, 99)
    recon_99p = np.percentile(recon_metric, 99)
    max_val = max(raw_99p, recon_99p)
    
    valid_mask = (raw_metric <= max_val) & (recon_metric <= max_val)
    raw_values_valid = raw_metric[valid_mask]
    recon_values_valid = recon_metric[valid_mask]
    
    # Calculate difference and average
    diff = recon_values_valid - raw_values_valid
    avg = (recon_values_valid + raw_values_valid) / 2
    
    # Calculate statistics for Bland-Altman plot
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    ax.scatter(avg, diff, alpha=0.5)
    
    # Add reference lines
    ax.axhline(mean_diff, color='k', linestyle='-', label=f'Mean: {mean_diff:.3f}')
    ax.axhline(upper_limit, color='r', linestyle='--', 
               label=f'+1.96 SD: {upper_limit:.3f}')
    ax.axhline(lower_limit, color='r', linestyle='--', 
               label=f'-1.96 SD: {lower_limit:.3f}')
    
    ax.set_title(f'Bland-Altman Plot for {metric_name}')
    ax.set_xlabel(f'Average of Raw and Reconstructed {metric_name}')
    ax.set_ylabel(f'Difference (Reconstructed - Raw)')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_{metric_name.lower().replace(' ', '_')}_bland_altman.png", dpi=300)
    plt.close(fig)

def plot_rq_metric_relative_difference_violin(raw_metrics, recon_metrics, plots_dir):
    """
    Create violin plot showing the relative difference distribution of all metrics.
    
    Parameters:
    -----------
    raw_metrics : dict
        Dictionary of raw waveform metrics
    recon_metrics : dict
        Dictionary of reconstructed waveform metrics
    plots_dir : str
        Directory to save plot
    """
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    metric_names = ['Pulse Area', 'Rise Time', 'Fall Time', 'FWHM', 'Peak Amplitude', 'Prompt Fraction', '10-90% Width (ns)']
    
    # Prepare data for violin plot
    diff_data = []
    
    for metric in metrics:
        raw_values = raw_metrics[metric]
        recon_values = recon_metrics[metric]
        
        # Create mask for valid points (non-zero raw values)
        valid_mask = (raw_values > 0)
        raw_values_valid = raw_values[valid_mask]
        recon_values_valid = recon_values[valid_mask]
        
        # Calculate relative differences (as percentages)
        rel_diff = 100 * (recon_values_valid - raw_values_valid) / raw_values_valid
        
        # Cap extreme values for better visualization
        rel_diff = np.clip(rel_diff, -100, 100)
        
        diff_data.append(rel_diff)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create violin plot
    violin_parts = ax.violinplot(diff_data, showmedians=True)
    
    # Set colors for violin plots
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add a horizontal line at y=0
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and grid
    ax.set_title('Relative Differences Between Raw and Reconstructed Metrics')
    ax.set_ylabel('Relative Difference (%)')
    ax.set_xticks(np.arange(1, len(metrics) + 1))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add text annotations for medians
    for i, data in enumerate(diff_data):
        median = np.median(data)
        ax.text(i + 1, np.max(data) + 5, f'Median: {median:.1f}%', 
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_relative_difference_violin.png", dpi=300)
    plt.close(fig)

def plot_rq_reconstruction_quality_radar(raw_metrics, recon_metrics, plots_dir):
    """
    Create a radar chart showing the reconstruction quality for each metric.
    
    Parameters:
    -----------
    raw_metrics : dict
        Dictionary of raw waveform metrics
    recon_metrics : dict
        Dictionary of reconstructed waveform metrics
    plots_dir : str
        Directory to save plot
    """
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    metric_names = ['Pulse Area', 'Rise Time', 'Fall Time', 'FWHM', 'Peak Amplitude', 'Prompt Fraction', '10-90% Width (ns)']
    
    # Calculate correlation and quality scores for each metric
    correlations = []
    quality_scores = []
    
    for metric in metrics:
        raw_values = raw_metrics[metric]
        recon_values = recon_metrics[metric]
        
        # Create mask for valid points
        valid_mask = (raw_values > 0)
        raw_values_valid = raw_values[valid_mask]
        recon_values_valid = recon_values[valid_mask]
        
        # Calculate correlation
        corr = np.corrcoef(raw_values_valid, recon_values_valid)[0, 1]
        correlations.append(corr)
        
        # Calculate mean fractional error
        fractional_errors = (recon_values_valid - raw_values_valid) / raw_values_valid
        fractional_errors = fractional_errors[np.abs(fractional_errors) <= 1.0]  # Remove extreme outliers
        mean_frac_error = np.mean(np.abs(fractional_errors))
        
        # Calculate quality score (higher is better)
        quality_score = abs(corr) * (1 - min(mean_frac_error, 1.0))
        quality_scores.append(quality_score)
    
    # Create radar chart
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric (equally spaced around the circle)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Make the plot close by repeating the first value
    values = quality_scores + [quality_scores[0]]
    angles = angles + [angles[0]]
    metric_names = metric_names + [metric_names[0]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, label='Quality Score')
    ax.fill(angles, values, alpha=0.25)
    
    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names[:-1])
    
    # Draw axis lines for each angle and label
    ax.set_rlabel_position(0)
    ax.grid(True)
    
    # Set y-limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    plt.title('Reconstruction Quality by Metric', size=15, y=1.1)
    
    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_quality_radar.png", dpi=300)
    plt.close(fig)

def plot_rq_distributions(raw_metrics, recon_metrics, plots_dir):
    """
    Plot distributions of various reconstruction quality metrics.
    
    Parameters:
    -----------
    raw_metrics : dict
        Dictionary of raw waveform metrics
    recon_metrics : dict
        Dictionary of reconstructed waveform metrics
    plots_dir : str
        Directory to save plots
    """
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    titles = ['Pulse Area (phd)', '10-90% Rise Time (ns)', '90-10% Fall Time (ns)', 
              'FWHM (ns)', 'Peak Amplitude (phd-ns)', 'Prompt Fraction', '10-90% Width (ns)']
    
    # Create individual plots for each metric
    for metric, title in zip(metrics, titles):
        plot_rq_metric_comparison(raw_metrics[metric], recon_metrics[metric], title, plots_dir)
        plot_rq_metric_fractional_error(raw_metrics[metric], recon_metrics[metric], title, plots_dir)
        plot_rq_metric_qq(raw_metrics[metric], recon_metrics[metric], title, plots_dir)
        plot_rq_metric_bland_altman(raw_metrics[metric], recon_metrics[metric], title, plots_dir)
    
    # Create combined plots
    plot_rq_metric_relative_difference_violin(raw_metrics, recon_metrics, plots_dir)
    plot_rq_reconstruction_quality_radar(raw_metrics, recon_metrics, plots_dir)

def plot_rq_correlation(raw_metrics, recon_metrics, plots_dir):
    """
    Update the correlation function to use the new modular approach.
    """
    # This functionality is now handled by the individual plot_rq_metric_comparison functions
    # We'll keep this function as a wrapper for backward compatibility
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    titles = ['Pulse Area (phd)', '10-90% Rise Time (ns)', '90-10% Fall Time (ns)', 
              'FWHM (ns)', 'Peak Amplitude (phd-ns)', 'Prompt Fraction', '10-90% Width (ns)']
    
    # Create a correlation heatmap instead of individual plots
    corr_matrix = np.zeros((len(metrics), 2))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        raw_values = raw_metrics[metric]
        recon_values = recon_metrics[metric]
        
        # Create mask for valid points
        valid_mask = (raw_values > 0)
        raw_values_valid = raw_values[valid_mask]
        recon_values_valid = recon_values[valid_mask]
        
        # Calculate auto-correlations and cross-correlation
        raw_auto = np.corrcoef(raw_values_valid, raw_values_valid)[0, 1]
        cross_corr = np.corrcoef(raw_values_valid, recon_values_valid)[0, 1]
        
        corr_matrix[i, 0] = raw_auto
        corr_matrix[i, 1] = cross_corr
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='viridis', vmin=0, vmax=1)
    
    # Set tick labels
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(['Raw Auto-correlation', 'Raw-Recon Correlation'])
    ax.set_yticklabels(titles)
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation Coefficient', rotation=-90, va="bottom")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(metrics)):
        for j in range(2):
            text = ax.text(j, i, f"{corr_matrix[i, j]:.3f}",
                          ha="center", va="center", color="w" if corr_matrix[i, j] < 0.7 else "black")
    
    ax.set_title("Correlation Analysis of Pulse Metrics")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_correlation_heatmap.png", dpi=300)
    plt.close(fig)

def plot_rq_fractional_errors(raw_metrics, recon_metrics, plots_dir):
    """
    Update the fractional errors function to use the new modular approach.
    """
    # This functionality is now handled by the individual plot_rq_metric_fractional_error functions
    # We'll keep this function for backward compatibility, but make it create a combined boxplot
    metrics = ['pulse_area', 'rise_time', 'fall_time', 'fwhm', 'peak_amplitude', 'prompt_fraction', 'width_1090']
    titles = ['Pulse Area', 'Rise Time', 'Fall Time', 'FWHM', 'Peak Amplitude', 'Prompt Fraction', '10-90% Width (ns)']
    
    # Prepare data for boxplot
    error_data = []
    
    for metric in metrics:
        raw_values = raw_metrics[metric]
        recon_values = recon_metrics[metric]
        
        # Create mask for valid points (non-zero raw values)
        valid_mask = (raw_values > 0)
        raw_values_valid = raw_values[valid_mask]
        recon_values_valid = recon_values[valid_mask]
        
        # Calculate fractional errors
        fractional_errors = (recon_values_valid - raw_values_valid) / raw_values_valid
        
        # Remove extreme outliers (beyond ±100%)
        valid_errors = fractional_errors[np.abs(fractional_errors) <= 1.0]
        
        error_data.append(valid_errors)
    
    # Create boxplot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    box = ax.boxplot(error_data, patch_artist=True, labels=titles, showfliers=False)
    
    # Color boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add a horizontal line at y=0
    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    
    ax.set_title('Fractional Errors Summary')
    ax.set_ylabel('(Reconstructed - Raw) / Raw')
    ax.set_xticklabels(titles, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/rq_fractional_errors_boxplot.png", dpi=300)
    plt.close(fig)


def main():
    # X, XC, C, PXC, EXC = load_SS_dataset('../dSSdMS/dSS_20241117_gaussgass_700samplearea7000_1.0e+04events_random_centered.npz')
    
    # train_split = int(0.9 * X.shape[0])
    
    # N = 4

    params = Params()
    
    # batch_size = 64
    # data_generator = N_channel_scatter_events_autoencoder_generator(XC[:train_split], max_N=N, batch_size=batch_size)
    # validation_data_generator = N_channel_scatter_events_autoencoder_generator(XC[train_split:], max_N=N, batch_size=batch_size)
    
    input_shape = (params.R * params.C, params.T)
    latent_dim = 128
    num_embeddings = 128
    embedding_dim = 128
    commitment_cost = 0.20
    adjacency_matrix = create_grid_adjacency(params.R)
    
    autoencoder = Autoencoder(input_shape, latent_dim, encoder_layer_sizes=[256, 256], decoder_layer_sizes=[256, 256])
    variational_autoencoder = VariationalAutoencoder(input_shape, latent_dim, encoder_layer_sizes=[256, 256], decoder_layer_sizes=[256, 256])
    graph_variational_autoencoder = BatchGraphVariationalAutoencoder(input_shape, 32, encoder_layer_sizes=[700, 512, 128, 64], decoder_layer_sizes=[64, 128, 512, 700], adjacency_matrix=adjacency_matrix)
    # graph_deep_variational_autoencoder = GraphDeepVariationalAutoencoder(input_shape, latent_dim, encoder_layer_sizes=[700, 128], decoder_layer_sizes=[128, 700], adjacency_matrix=adjacency_matrix)
    vq_variational_autoencoder = VQVariationalAutoencoder(input_shape, latent_dim, 256, embedding_dim=embedding_dim, commitment_cost=commitment_cost, encoder_layer_sizes=[128, 128], decoder_layer_sizes=[256, 256])
    graph_vq_variational_autoencoder = GraphVQVariationalAutoencoder(input_shape, 700, num_embeddings, embedding_dim=700, commitment_cost=commitment_cost, adjacency_matrix=adjacency_matrix, encoder_layer_sizes=[700, 700], decoder_layer_sizes=[700, 700])
    # simple_vq_variational_autoencoder = SimpleGraphVQVariationalAutoencoder(input_shape, 700, num_embeddings, embedding_dim=700, commitment_cost=commitment_cost, adjacency_matrix=adjacency_matrix, encoder_layer_sizes=[700, 700], decoder_layer_sizes=[700, 700])

    models = [autoencoder]#, variational_autoencoder, vq_variational_autoencoder, graph_vq_variational_autoencoder]
    losses = [reconstruction_loss]#, vae_loss, vqvae_loss, vqvae_loss]
    optimizers = [tf.keras.optimizers.Adam(learning_rate=5e-4)]#, tf.keras.optimizers.Adam(learning_rate=5e-4), tf.keras.optimizers.Adam(learning_rate=5e-4), tf.keras.optimizers.Adam(learning_rate=5e-4)]
    
    # load_models_from_checkpoint(models, losses, optimizers, data_generator)

    # test_events, _ = next(iter(validation_data_generator))
    # test_event = test_events[0].reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    # convolved_test_event = gaussian_blur_3d(test_event[np.newaxis, :, :, :, np.newaxis], kernel_size=5, sigma=1)

    _, e, _ = simulate_vertex_electron_photon_explicit([4], params)
    _, e_, _, _, _ = generate_channel_pulse(24, 24)
    plot_3d_scatter_with_profiles(e, 'Scatter', t_group_size=10)
    # plot_3d_scatter_with_profiles(convolved_test_event, 'Convolved Scatter', t_group_size=10)

    # plot_3d_multi_view(XC[0] + XC[1], 'Scatter')
    # plot_3d_multi_view(convolved_test_event, 'Convolved Scatter')

    # likelihood_test(autoencoder, samples=256, tune=True, loss=reconstruction_loss)
    rq_analysis(autoencoder, samples=4096, tune=True, loss=reconstruction_loss)

    e_flat = e.reshape((1, e.shape[0] * e.shape[1], e.shape[2]))
    reconstructed_e = autoencoder(e_flat)[0].numpy().reshape((params.R, params.C, params.T))
    plot_3d_scatter_with_profiles(reconstructed_e, 'Reconstructed Scatter', t_group_size=10)

    return

    train_models(models, losses, optimizers, data_generator, validation_data_generator, batch_size=batch_size, epochs=30, steps_per_epoch=64, use_checkpoints=False)
    
    test_events, _ = next(iter(validation_data_generator))
    
    test_event = test_events[0].reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    convolved_test_event = gaussian_blur_3d(test_event[np.newaxis, :, :, :, np.newaxis], kernel_size=5, sigma=1)
    autoencoder_reconstruction = autoencoder(test_events)[0].numpy().reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    variational_autoencoder_reconstruction = variational_autoencoder(test_events)[0][0].numpy().reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    vq_variational_autoencoder_reconstruction = vq_variational_autoencoder(test_events)[0][0].numpy().reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    graph_vq_variational_autoencoder_reconstruction = graph_vq_variational_autoencoder(test_events)[0][0].numpy().reshape((XC.shape[1], XC.shape[2], XC.shape[3]))
    
    display_events = [test_event, convolved_test_event, autoencoder_reconstruction, variational_autoencoder_reconstruction, vq_variational_autoencoder_reconstruction, graph_vq_variational_autoencoder_reconstruction]
    display_events_titles = [
        'Original', 
        'Original (Gaussian blur, σ=1)', 
        f'Autoencoder\nData size reduction: {int(autoencoder.get_data_size_reducton())}x\nReconstruction loss (MSE): {reconstruction_loss(test_event, autoencoder_reconstruction).numpy() : .3f} phd^2', 
        f'Variational Autoencoder\nData size reduction: {int(variational_autoencoder.get_data_size_reducton())}x\nReconstruction loss (MSE): {reconstruction_loss(test_event, variational_autoencoder_reconstruction).numpy() : .3f} phd^2', 
        f'Vector-Quantized Variational Autoencoder\nData size reduction: {int(vq_variational_autoencoder.get_data_size_reduction())}x\nReconstruction loss (MSE): {reconstruction_loss(test_event, vq_variational_autoencoder_reconstruction).numpy() : .3f} phd^2',
        f'Graph Vector-Quantized Variational Autoencoder\nData size reduction: {int(graph_vq_variational_autoencoder.get_data_size_reduction())}x\nReconstruction loss (MSE): {reconstruction_loss(test_event, graph_vq_variational_autoencoder_reconstruction).numpy() : .3f} phd^2'
    ]
    
    plot_events(np.stack(display_events, axis=0), title='', subtitles=display_events_titles)
    
    '''
    aux_data_generator = N_channel_scatter_events_generator(XC[:train_split], C[:train_split], PXC[:train_split], max_N=N, batch_size=batch_size, y='N')
    aux_val_data_generator = N_channel_scatter_events_generator(XC[train_split:], C[train_split:], PXC[train_split:], max_N=N, batch_size=batch_size, y='N')
    adjacency_matrix = create_grid_adjacency(XC.shape[1])
    
    def aux_model_build_compile(modelType, args, input_shape, loss=tf.keras.losses.MeanSquaredError(), metrics=[]):
        model = modelType(*args)
        model.build(input_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), 
            loss=loss,
            metrics=metrics
        )
        return model
    
    raw_aux_model_1 = aux_model_build_compile(GATNumScattersModel, [adjacency_matrix], (None, XC.shape[1], XC.shape[2]))
    compressed_aux_model_1 = aux_model_build_compile(DenseNumScattersModel, [], (None, latent_dim))
    compressed_aux_model_2 = aux_model_build_compile(DenseNumScattersModel, [], (None, latent_dim))
    compressed_aux_model_3 = aux_model_build_compile(GATNumScattersModel, [adjacency_matrix], (None, XC.shape[1], latent_dim))
    
    run_aux_task(
        [raw_aux_model_1, compressed_aux_model_1, compressed_aux_model_3],
        [None, autoencoder.compress, vq_variational_autoencoder.encode_to_codebook_vectors],
        labels = ['Raw data', 'AE latent', 'VQ-VAE codebook vectors'],
        data_generator=aux_data_generator, 
        val_data_generator=aux_val_data_generator,
        fname_suffix='_models'
    )
    
    aux_data_generator = N_channel_scatter_events_generator(XC[:train_split], C[:train_split], PXC[:train_split], max_N=N, batch_size=batch_size, y='XYZ')
    aux_val_data_generator = N_channel_scatter_events_generator(XC[train_split:], C[train_split:], PXC[train_split:], max_N=N, batch_size=batch_size, y='XYZ')
    
    loss = GATMultivariateNormalModel.combined_loss
    metrics = [
        GATMultivariateNormalModel.pdf_loss,
        GATMultivariateNormalModel.mask_loss,
    ]
    
    raw_aux_model_1 = aux_model_build_compile(GATMultivariateNormalModel, [adjacency_matrix], (None, XC.shape[1], XC.shape[2]), loss=loss, metrics=metrics)
    compressed_aux_model_1 = aux_model_build_compile(DenseMultivariateNormalModel, [], (None, latent_dim), loss=loss, metrics=metrics)
    compressed_aux_model_2 = aux_model_build_compile(DenseMultivariateNormalModel, [], (None, latent_dim), loss=loss, metrics=metrics)
    compressed_aux_model_3 = aux_model_build_compile(GATMultivariateNormalModel, [adjacency_matrix], (None, XC.shape[1], latent_dim), loss=loss, metrics=metrics)
    
    run_aux_task(
        [raw_aux_model_1, compressed_aux_model_1, compressed_aux_model_2, compressed_aux_model_3],
        [None, autoencoder.compress, vq_variational_autoencoder.compress, vq_variational_autoencoder.encode_to_codebook_vectors],
        labels = ['Raw data', 'AE latent', 'VQ-VAE codebook indices', 'VQ-VAE codebook vectors'],
        data_generator=aux_data_generator, 
        val_data_generator=aux_val_data_generator,
        fname_suffix='_models'
    )
    
    raw_aux_model_1 = aux_model_build_compile(GATMultivariateNormalModel, [adjacency_matrix], (None, XC.shape[1], XC.shape[2]), loss=loss, metrics=metrics)
    raw_aux_model_2 = aux_model_build_compile(GATMultivariateNormalModel, [adjacency_matrix], (None, XC.shape[1], XC.shape[2]), loss=loss, metrics=metrics)
    compressed_aux_model_1 = aux_model_build_compile(DenseMultivariateNormalModel, [], (None, latent_dim), loss=loss, metrics=metrics)
    
    run_aux_task(
        [raw_aux_model_1, compressed_aux_model_1, raw_aux_model_2],
        [None, autoencoder.encode, lambda x: autoencoder.decode(autoencoder.encode(x))],
        labels = ['Raw data', 'AE latent', 'AE reconstruction'],
        data_generator=aux_data_generator,
        val_data_generator=aux_val_data_generator,
        fname_suffix='_single_model'
    )
    
    '''
    X_test, XC_test, Y_test, C_test, P_test, E_test = generate_N_channel_scatter_events(X, XC, C, PXC, EXC, max_N=4, num_events=32768, normalize=True)
    
    # codebook_usage_histogram(vq_variational_autoencoder, XC_test[:256])
    
    fit_model = TSNE(n_components=2, perplexity=250, random_state=42)

    latent = np.concatenate([autoencoder.compress(XC_test[i:i + 256]).numpy() for i in tqdm(range(0, len(XC_test), 256))], axis=0)
    latent_space = fit_model.fit_transform(latent)
    
    vis_latent_space_footprint(fit_model, latent_space, XC_test)
    vis_latent_space_num_scatters(fit_model, latent_space, Y_test, N=4)
    vis_latent_space_phd(fit_model, latent_space, XC_test)
    
    
    
    
    
        
    

if __name__ == '__main__':
    set_mpl_style()
    main()