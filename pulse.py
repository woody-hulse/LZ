import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool
import time

from preprocessing import *

# Parameters (in ns)
DIFFWIDTH   = 300
G2          = 47.35
EGASWIDTH   = 450
PHDWIDTH    = 20
SAMPLERATE  = 10
ELECTRONS   = 148

def gaussian(x, C, mu, sigma):
    return C*np.exp(-(x-mu)**2/(2*sigma**2))

def generate_channel_pulse(num_rows, num_cols, mu=350, size=700, num_electrons=ELECTRONS // 2):
    summed_pulse = np.zeros(size)
    pulse = np.zeros((num_rows, num_cols, size))

    photon_interval_width = PHDWIDTH // SAMPLERATE * 3
    photon_interval = np.array([i for i in range(-photon_interval_width, photon_interval_width + 1)])
    phd_sample_width = PHDWIDTH / SAMPLERATE

    electron_arrival_times = np.random.normal(mu, DIFFWIDTH / SAMPLERATE, size=num_electrons)
    num_photons = np.random.poisson(G2, size=num_electrons)
    for e in range(num_electrons):
        # r = int(np.random.normal(num_cols / 2, num_cols / 8))
        # c = int(np.random.normal(num_rows / 2, num_rows / 8))
        photon_arrival_times = np.random.normal(electron_arrival_times[e], EGASWIDTH / SAMPLERATE, num_photons[e])
        for p in range(num_photons[e]):
            # Not considering multiple photoelectron emission
            # num_photoelectrons = np.abs(np.random.normal(1, 0.4))
            photon_indices = int(photon_arrival_times[p]) + photon_interval
            valid_indices = (photon_indices >= 0) & (photon_indices < size)
            summed_pulse[photon_indices[valid_indices]] += gaussian(photon_indices[valid_indices], 1, photon_arrival_times[p], phd_sample_width)

    # electron_arrival_times.sort()

    return summed_pulse, electron_arrival_times

        
def generate_ms_pulse(delta_mu=100):
    diff = (delta_mu / SAMPLERATE) // 2
    summed_pulse_1, electron_arrival_times_1 = generate_channel_pulse(num_rows=50, num_cols=50, mu=350 - diff, size=700)
    summed_pulse_2, electron_arrival_times_2 = generate_channel_pulse(num_rows=50, num_cols=50, mu=350 + diff, size=700)

    ms_pulse = summed_pulse_1 + summed_pulse_2
    electron_arrival_times = np.concatenate([electron_arrival_times_1, electron_arrival_times_2], axis=0)

    return ms_pulse, electron_arrival_times


def plot_pulses(pulses):
    summed_pulses = np.sum(pulses, axis=(0, 1))
    plt.plot(summed_pulses)
    plt.show()


def generate_ms_pulse_dataset(num_pulses, bins=20, max_delta_mu=1000, arrival_times=True, save=False):
    pulses = []
    electron_arrival_times = []
    delta_mu = []

    bin_sizes = num_pulses // bins

    debug_print(['generating dataset'])
    start_time = time.time()
    for i in tqdm(range(bins)):
        dmu = max_delta_mu // bins * i
        for j in range(bin_sizes):
            pulse, eats = generate_ms_pulse(delta_mu=dmu)
            pulses.append(np.expand_dims(pulse, axis=0))
            eats.sort()
            electron_arrival_times.append(np.expand_dims(eats, axis=0))
            delta_mu.append(dmu)
    end_time = time.time()
    debug_print(['Single processor dataset generation time:', end_time - start_time, 's'])

    pulses = np.concatenate(pulses, axis=0)
    delta_mu = np.array(delta_mu)
    electron_arrival_times = np.concatenate(electron_arrival_times, axis=0)

    if save: save_ms_pulse_dataset(pulses, delta_mu, electron_arrival_times, arrival_times)

    if arrival_times: return pulses, delta_mu, electron_arrival_times
    else: return pulses, delta_mu


def worker_task(dmu, bin_sizes):
    pulses = []
    electron_arrival_times = []
    for _ in range(bin_sizes):
        pulse, eats = generate_ms_pulse(delta_mu=dmu)
        pulses.append(np.expand_dims(pulse, axis=0))
        eats.sort()
        electron_arrival_times.append(np.expand_dims(eats, axis=0))
    return pulses, electron_arrival_times, [dmu] * bin_sizes


def generate_ms_pulse_dataset_multiproc(num_pulses, bins=20, max_delta_mu=1000, arrival_times=True, save=False):
    bin_sizes = num_pulses // bins
    tasks = [(max_delta_mu // bins * i, bin_sizes) for i in range(bins)]

    debug_print(['generating dataset'])
    start_time = time.time()
    with Pool() as pool:
        results = pool.starmap(worker_task, tasks)

    pulses = []
    electron_arrival_times = []
    delta_mu = []
    for result in results:
        pulses += result[0]
        electron_arrival_times += result[1]
        delta_mu += result[2]
    end_time = time.time()
    debug_print(['Multiprocessor dataset generation time:', end_time - start_time, 's'])

    pulses = np.concatenate(pulses, axis=0)
    delta_mu = np.array(delta_mu)
    electron_arrival_times = np.concatenate(electron_arrival_times, axis=0)

    if save: save_ms_pulse_dataset(pulses, delta_mu, electron_arrival_times, arrival_times)

    if arrival_times: return pulses, delta_mu, electron_arrival_times
    else: return pulses, delta_mu


def save_ms_pulse_dataset(pulses, delta_mu, electron_arrival_times, arrival_times):
    debug_print(['saving dataset'])
    num_pulses = pulses.shape[0]
    e = '{:.1e}'.format(num_pulses)
    weat = '_withEAT' if arrival_times else ''
    fname = f'../dSSdMS/dMS_2400320_gaussgass_700samplearea7000_areafrac0o5_deltamuinterval50ns_{e}events_random_centered{weat}.npz'
    np.savez_compressed(
        file=fname,
        events=pulses,
        delta_mu=delta_mu,
        electron_arrival_times=electron_arrival_times
    )
    debug_print(['saved dataset to', fname])


def load_ms_pulse_dataset(file):
    debug_print(['loading dataset from', file])
    with np.load(file) as f:
        pulses = f['events']
        delta_mu = f['delta_mu']
        electron_arrival_times = f['electron_arrival_times']
    
    return pulses, delta_mu, electron_arrival_times



def main():
    for i in range(20):
        generate_ms_pulse(delta_mu=i*5)
    
    # plot_pulses(pulses)

if __name__ == '__main__':
    main()