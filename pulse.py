import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from multiprocessing import Pool
import time

from preprocessing import *

# Parameters
DIFFWIDTH       = 300 # ns
G2              = 47.35
EGASWIDTH       = 20 # 450   # ns
PHDWIDTH        = 10  # ns
SAMPLERATE      = 10  # ns
NUM_ELECTRONS   = 148

def gaussian(x, C, mu, sigma):
    return C*np.exp(-(x-mu)**2/(2*sigma**2))

def generate_channel_pulse(num_rows, num_cols, mu=350, size=700, num_electrons=NUM_ELECTRONS // 2):
    summed_pulse    = np.zeros(size, dtype=np.float16)
    pulse           = np.zeros((num_rows, num_cols, size), dtype=np.float16)
    photon_pulse    = np.zeros((num_rows, num_cols, size), dtype=np.int8)
    electron_pulse  = np.zeros((num_rows, num_cols, size), dtype=np.int8)

    num_electrons = np.random.randint(0, 201) # 0 to 201 electrons

    photon_interval_width = PHDWIDTH // SAMPLERATE * 3
    photon_interval = np.array([i for i in range(-photon_interval_width, photon_interval_width + 1)])
    phd_sample_width = PHDWIDTH / SAMPLERATE

    electron_pulses = np.random.normal(mu, DIFFWIDTH / SAMPLERATE, size=num_electrons)
    num_photons = np.random.poisson(G2, size=num_electrons)
    r = np.random.normal(num_cols / 2, num_cols / 8, size=num_electrons) # Row of electron centers
    c = np.random.normal(num_rows / 2, num_rows / 8, size=num_electrons) # Column of electron centers
    for e in range(num_electrons):
        er, ec = min(num_cols - 1, max(0, int(r[e]))), min(num_rows - 1, max(0, int(c[e])))
        electron_pulse[er][ec][int(electron_pulses[e])] += 1
        pr = np.random.normal(r[e], num_cols / 32, size=num_photons[e]) # Row of photon centers
        pc = np.random.normal(c[e], num_cols / 32, size=num_photons[e]) # Column of photon centers
        photon_arrival_times = np.random.normal(electron_pulses[e], EGASWIDTH / SAMPLERATE, num_photons[e]) # Arrival times of photons
        for p in range(num_photons[e]):
            # Not considering multiple photoelectron emission
            num_photoelectrons = 1 # np.abs(np.random.normal(1, 0.4))
            
            photon_index = int(photon_arrival_times[p])
            photon_indices = photon_index + photon_interval
            valid_indices = (photon_indices >= 0) & (photon_indices < size)
            photon_emission = gaussian(photon_indices[valid_indices], 1, photon_arrival_times[p], phd_sample_width) * num_photoelectrons # Binned photon emission
            summed_pulse[photon_indices[valid_indices]] += photon_emission
            pri, pci = min(num_cols - 1, max(0, int(pr[p]))), min(num_rows - 1, max(0, int(pc[p])))
            pulse[pri][pci][photon_indices[valid_indices]] += photon_emission
            photon_pulse[pri][pci][photon_index] += 1

    return summed_pulse, pulse, photon_pulse, electron_pulse

def generate_pulse(mu=350, size=700, num_electrons=NUM_ELECTRONS // 2):
    summed_pulse = np.zeros(size)

    photon_interval_width = PHDWIDTH // SAMPLERATE * 3
    photon_interval = np.array([i for i in range(-photon_interval_width, photon_interval_width + 1)])
    phd_sample_width = PHDWIDTH / SAMPLERATE

    electron_pulses = np.random.normal(mu, DIFFWIDTH / SAMPLERATE, size=num_electrons)
    num_photons = np.random.poisson(G2, size=num_electrons)
    # num_photons = np.ones(num_electrons, dtype=np.int32)
    for e in range(num_electrons):
        # photon_arrival_times = np.random.normal(electron_pulses[e], EGASWIDTH / SAMPLERATE, num_photons[e])
        for p in range(num_photons[e]):
            # Not considering multiple photoelectron emission
            # num_photoelectrons = np.abs(np.random.normal(1, 0.4))
            photon_indices = int(electron_pulses[e]) + photon_interval
            valid_indices = (photon_indices >= 0) & (photon_indices < size)
            summed_pulse[photon_indices[valid_indices]] += gaussian(photon_indices[valid_indices], 1, electron_pulses[e], phd_sample_width)

    # electron_pulses.sort()
    return summed_pulse, None, None, electron_pulses


def generate_binary_channel_pulse(num_rows, num_cols, mu=350, size=700, num_electrons=NUM_ELECTRONS):
    summed_pulse = np.zeros(size, dtype=np.int8)
    pulse = np.zeros((num_rows, num_cols, size), dtype=np.int8)

    electron_pulses = np.random.normal(mu, DIFFWIDTH / SAMPLERATE, size=num_electrons)
    # num_photons = np.random.poisson(G2, size=num_electrons)
    num_photons = np.ones(num_electrons, dtype=np.int32)
    for e in range(num_electrons):
        r = int(np.random.normal(num_cols / 2, num_cols / 8))
        c = int(np.random.normal(num_rows / 2, num_rows / 8))
        r = min(num_cols - 1, max(0, r))
        c = min(num_rows - 1, max(0, c))
        photon_arrival_times = np.random.normal(electron_pulses[e], EGASWIDTH / SAMPLERATE, num_photons[e])
        for p in range(num_photons[e]):
            # Not considering multiple photoelectron emission
            # num_photoelectrons = np.abs(np.random.normal(1, 0.4))
            photon_index = int(photon_arrival_times[p])
            summed_pulse[photon_index] = 1
            pulse[r][c][photon_index] = 1

    # electron_pulses.sort()

    return summed_pulse, pulse, None, electron_pulses

        
def generate_ms_pulse(delta_mu=100):
    diff = (delta_mu / SAMPLERATE) // 2
    summed_pulse_1, electron_pulses_1 = generate_pulse(mu=350 - diff, size=700)
    summed_pulse_2, electron_pulses_2 = generate_pulse(mu=350 + diff, size=700)

    ms_pulse = summed_pulse_1 + summed_pulse_2
    electron_pulses = np.concatenate([electron_pulses_1, electron_pulses_2], axis=0)

    return ms_pulse, electron_pulses


def plot_pulses(pulses):
    summed_pulses = np.sum(pulses, axis=(0, 1))
    plt.plot(summed_pulses)
    plt.show()


def generate_pulse_dataset(num_pulses, bins=20, max_delta_mu=1000, arrival_times=True, save=False):
    pulses = []
    electron_pulses = []
    delta_mu = []

    bin_sizes = num_pulses // bins

    debug_print(['generating dataset'])
    start_time = time.time()
    for i in tqdm(range(bins)):
        dmu = max_delta_mu // bins * i
        for j in range(bin_sizes):
            pulse, electron_pulse = generate_ms_pulse(delta_mu=dmu)
            pulses.append(pulse)
            electron_pulses.append(electron_pulse)
            delta_mu.append(dmu)
    end_time = time.time()
    debug_print(['Single processor dataset generation time:', end_time - start_time, 's'])

    pulses = np.array(pulses)
    delta_mu = np.array(delta_mu)
    electron_pulses = np.array(electron_pulses)

    if save: save_pulse_dataset(pulses, delta_mu, electron_pulses, arrival_times)

    if arrival_times: return pulses, delta_mu, electron_pulses
    else: return pulses, delta_mu

def pulse_task(dmu, bin_sizes, num_electrons):
    pulses = []
    summed_pulses = []
    photon_pulses = []
    electron_pulses = []
    for _ in range(bin_sizes):
        summed_pulse, pulse, photon_pulse, electron_pulse = generate_pulse(num_electrons=num_electrons)
        pulses.append(pulse)
        photon_pulses.append(photon_pulse)
        summed_pulses.append(summed_pulse)
        electron_pulses.append(electron_pulse)
    return summed_pulses, pulses, photon_pulses, electron_pulses, [dmu] * bin_sizes

def channel_pulse_task(dmu, bin_sizes, num_electrons):
    pulses = []
    summed_pulses = []
    photon_pulses = []
    electron_pulses = []
    for _ in range(bin_sizes):
        summed_pulse, pulse, photon_pulse, electron_pulse = generate_channel_pulse(num_rows=16, num_cols=16, num_electrons=num_electrons)
        pulses.append(pulse)
        summed_pulses.append(summed_pulse)
        photon_pulses.append(photon_pulse)
        electron_pulses.append(electron_pulse)
    return summed_pulses, pulses, photon_pulses, electron_pulses, [dmu] * bin_sizes

def ms_channel_pulse_task(dmu, bin_sizes, num_electrons):
    pulses = []
    summed_pulses = []
    photon_pulses = []
    electron_pulses = []
    for _ in range(bin_sizes):
        diff = (dmu / SAMPLERATE) // 2
        summed_pulse_1, pulse_1, photon_pulse_1, electron_pulse_1 = generate_channel_pulse(num_rows=16, num_cols=16, mu=350 - diff, size=700, num_electrons=num_electrons)
        summed_pulse_2, pulse_2, photon_pulse_2, electron_pulse_2 = generate_channel_pulse(num_rows=16, num_cols=16, mu=350 + diff, size=700, num_electrons=num_electrons)

        summed_pulse = summed_pulse_1 + summed_pulse_2
        pulse = pulse_1 + pulse_2
        photon_pulse = photon_pulse_1 + photon_pulse_2

        electron_pulse = np.concatenate([electron_pulse_1, electron_pulse_2], axis=0)
        electron_pulse.sort()
        pulses.append(pulse)
        summed_pulses.append(summed_pulse)
        photon_pulses.append(photon_pulse)
        electron_pulses.append(electron_pulse)

    return summed_pulses, pulses, photon_pulses, electron_pulses, [dmu] * bin_sizes

def binary_channel_pulse_task(dmu, bin_sizes, num_electrons):
    pulses = []
    summed_pulses = []
    photon_pulses = []
    electron_pulses = []
    for _ in range(bin_sizes):
        summed_pulse, pulse, photon_pulse, electron_pulse = generate_binary_channel_pulse(num_rows=10, num_cols=10, num_electrons=num_electrons)
        pulses.append(pulse)
        photon_pulses.append(photon_pulse)
        summed_pulses.append(summed_pulse)
        electron_pulse.sort()
        electron_pulses.append(electron_pulse)
    return summed_pulses, pulses, photon_pulses, electron_pulses, [dmu] * bin_sizes


def generate_pulse_dataset_multiproc(num_pulses, bins=20, max_delta_mu=1000, arrival_times=True, save=False, num_electrons=NUM_ELECTRONS, task=pulse_task):
    bin_sizes = num_pulses // bins
    tasks = [(max_delta_mu // bins * i, bin_sizes, num_electrons) for i in range(bins)]

    debug_print(['generating dataset'])
    start_time = time.time()
    with Pool(processes=multiprocessing.cpu_count()-2) as pool:
        results = pool.starmap(task, tasks)

    summed_pulses = []
    pulses = []
    photon_pulses = []
    electron_pulses = []
    delta_mu = []
    for result in results:
        summed_pulses += result[0]
        pulses += result[1]
        photon_pulses += result[2]
        electron_pulses += result[3]
        delta_mu += result[4]
    end_time = time.time()
    debug_print(['Multiprocessor dataset generation time:', end_time - start_time, 's'])

    summed_pulses = np.array(summed_pulses)
    pulses = np.array(pulses)
    photon_pulses = np.array(photon_pulses)
    delta_mu = np.array(delta_mu)
    electron_pulses = np.array(electron_pulses)

    if save: save_pulse_dataset(summed_pulses, pulses, photon_pulses, delta_mu, electron_pulses, arrival_times)

    if arrival_times: return summed_pulses, pulses, photon_pulses, delta_mu, electron_pulses
    else: return summed_pulses, pulses, delta_mu


def save_pulse_dataset(summed_pulses, pulses, photon_pulses, delta_mu, electron_pulses, arrival_times):
    debug_print(['saving dataset'])
    num_pulses = pulses.shape[0]
    e = '{:.1e}'.format(num_pulses)
    weat = '_withEAT' if arrival_times else ''
    fname = f'../dSSdMS/dSS_2400928_gaussgass_700samplearea7000_areafrac0o5_{e}events_random_centered.npz'
    np.savez_compressed(
        file=fname,
        events=summed_pulses,
        channel_events=pulses,
        photon_events=photon_pulses,
        delta_mu=delta_mu,
        electron_pulses=electron_pulses
    )
    debug_print(['saved dataset to', fname])


def load_pulse_dataset(file):
    debug_print(['loading dataset from', file])
    with np.load(file, allow_pickle=True) as f:
        summed_pulses = f['events']
        pulses = f['channel_events']
        photon_pulses = f['photon_events']
        delta_mu = f['delta_mu']
        electron_pulses = f['electron_pulses']
    
    return summed_pulses, pulses, photon_pulses, delta_mu, electron_pulses

def load_pulse_dataset_old(file):
    debug_print(['loading dataset from', file])
    with np.load(file, allow_pickle=True) as f:
        pulses = f['DSdata']
        dmu = f['UL_values'][:, 1]
    
    dprint(f'Loaded pulses with shape:   {pulses.shape}')
    dprint(f'Loaded delta mu with shape: {dmu.shape}')
    
    return pulses, dmu

def at_to_hist(at):
    at_hist = []
    debug_print(['generating arrival time histograms'])
    for times in tqdm(at):
        hist, bins = np.histogram(times, bins=np.arange(0, 701, 1))
        at_hist.append(hist)
    at_hist = np.array(at_hist)

    return at_hist

def plot_at_hist(hist, label):
    plt.fill_between(range(len(hist)), hist, alpha=0.3)
    plt.plot(hist, label=label)


def plot_3d_hist(hist2d, xedges=None, yedges=None):
    if xedges is None:
        xedges = np.arange(hist2d.shape[0] + 1)
    if yedges is None:
        yedges = np.arange(hist2d.shape[1] + 1)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist2d.ravel()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_xlabel('Channel')
    ax.set_ylabel('')
    ax.set_zlabel('Liquid Electrons')
    ax.set_title('Liquid electron count 2D CDF')
    plt.show()    



def main():
    summed_pulses, pulses, photon_pulses, delta_mu, electron_pulses = generate_pulse_dataset_multiproc(int(5e4), bins=20, max_delta_mu=0, arrival_times=True, save=True,
                                     
                                     task = channel_pulse_task
                                     
                                     )

    debug_print(['Generated summed pulses dataset with shape            :', summed_pulses.shape])
    debug_print(['Generated pulses dataset with shape                   :', pulses.shape])
    debug_print(['Generated delta mu dataset with shape                 :', delta_mu.shape])
    debug_print(['Generated photon pulses dataset with shape            :', photon_pulses.shape])
    debug_print(['Generated electron pulses dataset with shape          :', electron_pulses.shape])

if __name__ == '__main__':
    main()