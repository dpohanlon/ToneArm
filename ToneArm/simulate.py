import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

from tqdm import tqdm

import numpy as np

from scipy.io import wavfile

from scipy.signal import resample, savgol_filter

from Coil import Coil
from Stylus import Stylus
from Processing import low_pass_filter, riaa_filter, normalize_audio

import torch

def write_to_wav(filename, audio, sample_rate = 44100):
    """
    Write the audio signal to a WAV file.

    Parameters:
    filename (str): Output WAV file name.
    audio (numpy.array): Audio signal to write.
    sample_rate (int): Sampling rate of the audio signal.
    """
    # Convert to 16-bit integers for WAV file format
    int_audio = np.int16(audio * 32767)
    wavfile.write(filename, sample_rate, int_audio)

def filter_stylus_radius(signal, stylus, velocity = 0.5):

    cutoff_freq = velocity / (4 * stylus.radius)

    return low_pass_filter(signal, cutoff_freq)

def bump(freq, length):

    xs = np.linspace(0, length, length)

    return np.exp((- 10 ** -np.log(0.01 * length) * (xs - length /2) ** 2)) * np.sin(2 * np.pi * (freq / 44100) * xs)

if __name__ == '__main__':

    groove_width = 0.05E-3
    groove_pitch = 1E-3

    # 1 second, 0.36m

    freq = 440 # Hz

    coil = Coil(coil_radius=1E-2, number_of_turns=1000, remanence=1.0, magnet_volume= 0.01 * 0.01 * 0.01)

    ticks = 44100
    total_time = 10

    # particle_size = 50E-6 # 50 um
    particle_size = 0.5E-3 # 0.5 mm

    distance_per_second = 0.36

    ticks_per_metre = ticks / distance_per_second

    ticks_per_particle = int(ticks_per_metre * particle_size)

    print(ticks_per_metre, ticks_per_particle)

    # Fudge factor so that maximum deviation causes slight distortion at ~5mV peak-peak

    ticks, data = wavfile.read("/Users/dohanlon/Downloads/ff-16b-1c-44100hz.wav")

    print(r)

    # Mono
    # data = np.mean(data, axis = 1)

    data = data.astype('float')

    # 10 seconds
    data = data[:int(r * 10)]
    data = normalize_audio(data)
    data = groove_pitch * data

    write_to_wav('nothing.wav', normalize_audio(data))

    for s in np.random.randint(0, 200000, size = 50):

        l = np.random.choice([100, 200, 500, 1000], p = [0.45, 0.3, 0.2, 0.05])
        p = np.array([1, 1, 4, 4, 2, 1, 0.5])
        p /= np.sum(p)

        a = np.clip(np.random.normal(0.001, 0.001), 0, 0.002)

        data[s:s + l] += a * bump(freq = np.random.choice([10, 50, 100, 500, 1000, 2000, 5000], p = p), length = l)

    for s in np.random.randint(0, 200000, size = 200):

        l = 100
        a = np.clip(np.random.normal(1E-4, 1E-4), 0, 2E-4)
        f = np.random.randint(5000, 12000)

        data[s:s + l] += a * bump(freq = f, length = l)

    plt.plot(np.linspace(0, total_time, ticks)[:1000], data[:1000])
    plt.savefig('noise.pdf')
    plt.clf()

    plt.plot(savgol_filter(data[:1000], 10, 3))
    plt.savefig('savgol.pdf')
    plt.clf()

    deltaPos = data[1:] - data[:-1]

    data_in = data

    # data += np.random.normal(0, 1E-5, size = len(data))

    # data = normalize_audio(riaa_biquad(data))

    # data = riaa_filter(data, mode = 'recording')

    voltages = []
    fluxes = []
    for i in tqdm(range(len(data) - 2)):
        voltage, dFlux = coil.induced_voltage(initial_distance=data[i], final_distance=data[i + 1], time_interval=1/ticks)
        voltages.append(voltage)
        fluxes.append(dFlux)

    plt.plot(fluxes[100:1000])
    plt.savefig('f.pdf')
    plt.clf()

    voltages = np.array(voltages)

    voltages += np.random.normal(0, 1E-4, size = len(voltages))

    stylus = Stylus()

    voltages = riaa_filter(voltages, mode = 'playback')

    voltages_filtered = filter_stylus_radius(normalize_audio(voltages), stylus)

    # plt.plot(np.linspace(0, total_time, ticks)[:-2][500:1000], voltages[500:1000], label = 'V', lw = 1.0)
    plt.plot(np.linspace(0, total_time, ticks)[:-2][500:1000], voltages[500:1000], label = 'V', lw = 1.0)
    plt.plot(np.linspace(0, total_time, ticks)[:-2][500:1000], 0.001 * voltages_filtered[500:1000], label = 'Vf', lw = 1.0)
    plt.plot(np.linspace(0, total_time, ticks)[:-2][500:1000], 20 * data_in[501:1001], label = 'x', lw = 1.0)
    plt.legend(loc = 0)

    plt.savefig('v.pdf')
    plt.clf()

    plt.plot(np.linspace(0, total_time, ticks)[:-2][500:1000], 1000*voltages[500:1000], label = 'V', lw = 1.0)
    plt.plot(np.linspace(0, total_time, ticks)[:-2][500:1000], 50 * deltaPos[500:1000], label = 'dx', lw = 1.0)
    plt.legend(loc = 0)

    plt.savefig('dv.pdf')
    plt.clf()

    norm_voltages = normalize_audio(voltages)
    write_to_wav('test.wav', norm_voltages)

    voltages_filtered = filter_stylus_radius(norm_voltages, stylus)
    write_to_wav('test_filtered.wav', voltages_filtered)

    plt.plot(np.linspace(0, total_time, ticks)[:-2][:1000], voltages_filtered[:1000])

    plt.savefig('v_f.pdf')
    plt.clf()
