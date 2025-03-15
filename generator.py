import numpy as np
import tensorflow as tf

from preprocessing import *

def generate_N_scatter_events(SS, max_N=4, num_events=int(1e5)):
    num_samples = SS.shape[0]
    num_channels = SS.shape[1]
    center = num_channels / 2

    X = np.zeros((num_events, num_channels))
    Y = np.zeros((num_events, max_N))
    
    dprint('Generating N scatter events')
    for i in tqdm(range(num_samples)):
        n = np.random.randint(1, max_N + 1)
        mu_offsets = np.array(np.random.normal(0, 50, n), dtype=np.int32)
        mus = center + mu_offsets
        pulses_idx = np.random.choice(num_samples, n, replace=False)
        offset_pulses = [np.roll(SS[pulses_idx[j]], mu_offsets[j]) for j in range(n)]
        X[i] = np.sum(offset_pulses, axis=0)
        Y[i, :n] = np.sort(mus)
    
    return X, Y

def generate_N_channel_scatter_events(X, XC, C, PXC, EXC, max_N=4, num_events=int(1e5), normalize=True):
    num_samples = X.shape[0]
    num_channels = X.shape[1]
    num_pmt_rows = XC.shape[1]
    center = num_channels / 2

    P = np.sum(PXC, axis=(1, 2, 3))
    E = np.sum(EXC, axis=(1, 2, 3))

    X_ = np.zeros((num_events, num_channels), dtype=np.float16)
    XC_ = np.zeros((num_events, XC.shape[1], XC.shape[2], num_channels), dtype=np.float16)
    PXC_ = np.zeros((num_events, XC.shape[1], XC.shape[2], num_channels), dtype=np.uint8)
    Y_ = np.zeros((num_events, max_N, 1), dtype=np.float16)
    C_ = np.zeros((num_events, max_N, 2), dtype=np.float16)
    P_ = np.zeros((num_events, max_N, 1), dtype=np.float16)
    E_ = np.zeros((num_events, max_N, 1), dtype=np.float16)

    dprint('Generating N scatter events')
    for i in tqdm(range(num_events)):
        n = np.random.randint(1, max_N + 1)
        mu_offsets = np.array(np.random.normal(0, 50, n), dtype=np.int32)
        mus = center + mu_offsets
        pulses_idx = np.random.choice(num_samples, n, replace=False)
        offset_pulses = [np.roll(X[pulses_idx[j]], mu_offsets[j]) for j in range(n)]
        offset_channel_pulses = [np.roll(XC[pulses_idx[j]], mu_offsets[j], axis=-1) for j in range(n)]
        offset_channel_photon_pulses = [np.roll(PXC[pulses_idx[j]], mu_offsets[j], axis=-1) for j in range(n)]
        noise = np.random.normal(0, 0.1, size=(XC.shape[1], XC.shape[2], num_channels))

        time_sorted_pulses_indices = np.argsort(mus)

        X_[i] = np.sum(offset_pulses, axis=0)
        XC_[i] = np.sum(offset_channel_pulses, axis=0) + noise
        PXC_[i] = np.sum(offset_channel_photon_pulses, axis=0)
        Y_[i, :n] = np.expand_dims(mus[time_sorted_pulses_indices], axis=-1)
        C_[i, :n] = C[pulses_idx][time_sorted_pulses_indices]
        P_[i, :n] = np.expand_dims(P[pulses_idx][time_sorted_pulses_indices], axis=-1)
        E_[i, :n] = np.expand_dims(E[pulses_idx][time_sorted_pulses_indices], axis=-1)
    
    if normalize:
        XC_ = XC_.reshape((XC_.shape[0], XC_.shape[1] * XC_.shape[2], XC_.shape[3]))
        Y_ /= num_channels
        C_ /= num_pmt_rows
        P_ /= 10000

    return X_, XC_, Y_, C_, P_, E_

def N_channel_scatter_events_generator(XC, C, PXC, max_N=4, batch_size=256, y='XYZ'):
    num_samples = XC.shape[0]
    num_channels = XC.shape[3]
    num_pmt_rows = XC.shape[1]
    center = num_channels / 2

    P = np.sum(PXC, axis=(1, 2, 3))

    while True:
        XC_ = np.zeros((batch_size, XC.shape[1], XC.shape[2], num_channels), dtype=np.float32)
        # PXC_ = np.zeros((batch_size, XC.shape[1], XC.shape[2], num_channels), dtype=np.uint8)
        Y_ = np.zeros((batch_size, max_N, 1), dtype=np.float32)
        C_ = np.zeros((batch_size, max_N, 2), dtype=np.float32)
        P_ = np.zeros((batch_size, max_N, 1), dtype=np.float32)
        N = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            n = np.random.randint(1, max_N + 1)
            N[i] = n
            mu_offsets = np.array(np.random.normal(0, 50, n), dtype=np.int32)
            mus = center + mu_offsets
            pulses_idx = np.random.choice(num_samples, n, replace=False)
            offset_channel_pulses = [np.roll(XC[pulses_idx[j]], mu_offsets[j], axis=-1) for j in range(n)]
            # offset_channel_photon_pulses = [np.roll(PXC[pulses_idx[j]], mu_offsets[j], axis=-1) for j in range(n)]
            noise = np.random.normal(0, 0.1, size=(XC.shape[1], XC.shape[2], num_channels))

            time_sorted_pulses_indices = np.argsort(mus)

            XC_[i] = np.sum(offset_channel_pulses, axis=0) + noise
            # PXC_[i] = np.sum(offset_channel_photon_pulses, axis=0)
            Y_[i, :n] = np.expand_dims(mus[time_sorted_pulses_indices], axis=-1)
            C_[i, :n] = C[pulses_idx][time_sorted_pulses_indices]
            P_[i, :n] = np.expand_dims(P[pulses_idx][time_sorted_pulses_indices], axis=-1)

        XC_ = XC_.reshape((XC_.shape[0], XC_.shape[1] * XC_.shape[2], XC_.shape[3]))
        Y_ /= num_channels
        C_ /= num_pmt_rows
        P_ /= 10000
    
        XYZ_ = np.concatenate([C_, Y_, P_], axis=-1)

        if y == 'XYZ':
            yield XC_, XYZ_, P_
        elif y == 'N':
            yield XC_, N, P_
        

def N_channel_scatter_events_autoencoder_generator(XC, max_N=4, batch_size=256):
    num_samples = XC.shape[0]
    num_channels = XC.shape[3]
    num_pmt_rows = XC.shape[1]
    center = num_channels / 2

    while True:
        XC_ = np.zeros((batch_size, XC.shape[1], XC.shape[2], num_channels), dtype=np.float32)

        for i in range(batch_size):
            n = np.random.randint(1, max_N + 1)
            mu_offsets = np.array(np.random.normal(0, 50, n), dtype=np.int32)
            mus = center + mu_offsets
            pulses_idx = np.random.choice(num_samples, n, replace=False)
            offset_channel_pulses = [np.roll(XC[pulses_idx[j]], mu_offsets[j], axis=-1) for j in range(n)]
            noise = np.random.normal(0, 0.1, size=(XC.shape[1], XC.shape[2], num_channels))

            XC_[i] = np.sum(offset_channel_pulses, axis=0) + noise

        XC_ = XC_.reshape((XC_.shape[0], XC_.shape[1] * XC_.shape[2], XC_.shape[3]))

        yield XC_, XC_
       
        
def N_channel_scatter_events_generator_2(XC, C, PXC, max_N=4, batch_size=256):
    num_samples = XC.shape[0]
    num_pmt_rows = XC.shape[1]
    num_channels = XC.shape[3]
    center = num_channels / 2

    P = np.sum(PXC, axis=(1, 2, 3))
    pmt_rows, pmt_cols = XC.shape[1], XC.shape[2]

    while True:
        # Generate all random variables for the batch
        n_arr = np.random.randint(1, max_N + 1, size=batch_size)
        mu_offsets = np.random.normal(0, 50, (batch_size, max_N)).astype(np.int32)
        mus = center + mu_offsets

        # For no-replacement per sample, we do a small loop.
        # This is the only loop remaining and is minimal. If absolute no loops are required,
        # consider another sampling strategy that may slightly alter the distribution.
        pulses_idx = np.empty((batch_size, max_N), dtype=np.int32)
        for i in range(batch_size):
            pulses_idx[i] = np.random.choice(num_samples, max_N, replace=False)

        # Extract the pulses for all samples and pulses at once
        # XC[pulses_idx] has shape (batch_size, max_N, pmt_rows, pmt_cols, num_channels)
        selected_XC = XC[pulses_idx]
        selected_C = C[pulses_idx]      # (batch_size, max_N, 2)
        selected_P = P[pulses_idx]      # (batch_size, max_N)

        # Perform the rolling along the last axis
        length = num_channels
        indices = np.arange(length)
        shifts = mu_offsets % length  # (batch_size, max_N)

        # We'll use np.take_along_axis for rolling:
        # We need to construct the shifted indices for the last axis.
        # The shape for indexing: (batch_size, max_N, 1, 1, num_channels) broadcasted
        # We'll reshape selected_XC to (batch_size, max_N, pmt_rows*pmt_cols, num_channels) for easier indexing
        R = pmt_rows * pmt_cols
        reshaped_XC = selected_XC.reshape(batch_size, max_N, R, length)

        # Create rolled indices
        # rolled_indices[i,j] = (indices - shifts[i,j]) % length
        # Shape: (batch_size, max_N, length)
        rolled_indices = ((indices[None, None, :] - shifts[:, :, None]) % length)
        # Expand to match (batch_size, max_N, R, length) by broadcasting
        rolled_indices = np.broadcast_to(rolled_indices[:, :, None, :], (batch_size, max_N, R, length))

        # Apply the rolling
        rolled_data = np.take_along_axis(reshaped_XC, rolled_indices, axis=-1)
        # rolled_data shape: (batch_size, max_N, R, num_channels)
        # Reshape back to (batch_size, max_N, pmt_rows, pmt_cols, num_channels)
        rolled_data = rolled_data.reshape(batch_size, max_N, pmt_rows, pmt_cols, num_channels)

        # Sum over pulses
        summed_pulses = np.sum(rolled_data, axis=1)  # (batch_size, pmt_rows, pmt_cols, num_channels)

        # Add noise
        noise = np.random.normal(0, 0.1, size=(batch_size, pmt_rows, pmt_cols, num_channels))
        XC_ = summed_pulses + noise

        # Sort pulses by time (mus)
        time_sorted_indices = np.argsort(mus, axis=1)  # (batch_size, max_N)

        # Create output arrays
        Y_ = np.zeros((batch_size, max_N, 1), dtype=np.float32)
        C_ = np.zeros((batch_size, max_N, 2), dtype=np.float32)
        P_ = np.zeros((batch_size, max_N, 1), dtype=np.float32)

        # Reorder C and P according to sorted times
        sorted_mus = np.take_along_axis(mus, time_sorted_indices, axis=1)
        sorted_C = np.take_along_axis(selected_C, time_sorted_indices[:, :, None], axis=1)
        sorted_P = np.take_along_axis(selected_P[:, :, None], time_sorted_indices[:, :, None], axis=1)

        # Mask out unused pulses (n < max_N)
        mask = np.arange(max_N)[None, :] < n_arr[:, None]
        Y_[mask] = sorted_mus[mask, None]
        C_[mask] = sorted_C[mask]
        P_[mask] = sorted_P[mask]

        # Reshaping and normalization
        XC_ = XC_.reshape((batch_size, pmt_rows * pmt_cols, num_channels))
        Y_ /= num_channels
        C_ /= num_pmt_rows
        P_ /= 10000.0

        XYZ_ = np.concatenate([C_, Y_], axis=-1)

        yield XC_, XYZ_


def generate_double_scatter_events(X, XC, C, num_bins=20, max_dmu=1000, num_events=int(1e4)):
    num_samples = X.shape[0]
    num_channels = X.shape[1]
    center = num_channels / 2

    X_ = np.zeros((num_events, num_channels))
    XC_ = np.zeros((num_events, XC.shape[1], XC.shape[2], num_channels))
    Y_ = np.zeros((num_events, 4))
    C_ = np.zeros((num_events, 4, 2))
    DMU = np.zeros(num_events)

    dprint('Generating double scatter events')
    for i in tqdm(range(num_events)):
        n = 2
        dmu = (i % num_bins) * max_dmu / num_bins
        mu1 = int(center - dmu // 2)
        mu2 = int(center + dmu // 2)
        pulses_idx = np.random.choice(num_samples, 2, replace=False)
        offset_pulses = [np.roll(X[pulses_idx[j]], mu1) for j in range(2)]
        offset_channel_pulses = [np.roll(XC[pulses_idx[j]], mu1, axis=2) for j in range(2)]
        X_[i] = np.sum(offset_pulses, axis=0)
        XC_[i] = np.sum(offset_channel_pulses, axis=0)
        Y_[i, :n] = np.sort([mu1, mu2])
        C_[i, :n] = np.sort(np.array([C[pulses_idx[j]] for j in range(n)]), axis=0)
        DMU[i] = dmu
    
    return X_, XC_, Y_, C_, DMU

