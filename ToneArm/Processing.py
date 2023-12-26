import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

import numpy as np

import scipy.signal as signal

from tqdm import tqdm

import torch


def low_pass_filter(input_signal, cutoff_freq, sampling_rate=44100, order=5):

    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    filtered_signal = signal.filtfilt(b, a, input_signal)

    return filtered_signal


def riaa_coeffs():

    # 44.1 kHz sample rate
    zeros = [-0.2014898, 0.9233820]
    poles = [0.7083149, 0.9924091]

    # Numerator
    # polynomial coefficients with roots zeros[0] and zeros[1]
    b0 = 1.0
    b1 = -(zeros[0] + zeros[1])
    b2 = zeros[0] * zeros[1]

    # Denominator
    # polynomial coefficients with roots poles[0] and poles[1]
    a0 = 1.0
    a1 = -(poles[0] + poles[1])
    a2 = poles[0] * poles[1]

    # Normalize to 0dB at 1kHz
    y = 2 * np.pi * 1000 / 44100
    b_re = b0 + b1 * np.cos(-y) + b2 * np.cos(-2 * y)
    a_re = a0 + a1 * np.cos(-y) + a2 * np.cos(-2 * y)
    b_im = b1 * np.sin(-y) + b2 * np.sin(-2 * y)
    a_im = a1 * np.sin(-y) + a2 * np.sin(-2 * y)
    g = 1 / np.sqrt((b_re**2 + b_im**2) / (a_re**2 + a_im**2))

    b0 *= g
    b1 *= g
    b2 *= g

    a = [a0, a1, a2]
    b = [b0, b1, b2]

    return b, a


def riaa_filter(data, mode="playback"):

    b, a = riaa_coeffs()

    if mode == "playback":
        return signal.lfilter(b, a, data)
    else:
        return signal.lfilter(a, b, data)


def normalize_audio(audio, peak=1.0):
    """
    Normalize the audio signal to the given peak level.

    Parameters:
    audio (numpy.array): Input audio signal.
    peak (float): Peak value to normalize to. Default is 1.0 (maximum for float32).

    Returns:
    numpy.array: Normalized audio signal.
    """
    max_value = np.max(np.abs(audio))
    if max_value == 0:
        return audio  # Return original audio if it's silent to avoid division by zero
    return peak * audio / max_value


if __name__ == "__main__":

    b, a = riaa_coeffs()

    # Frequency response
    # w, h = signal.freqz(b, a, worN=8000) # Playback
    w, h = signal.freqz(a, b, worN=8000)  # Recording

    # Generate frequency axis in Hz (assuming a sampling rate of 44.1 kHz)
    fs = 44100
    freqs = w * fs / (2 * np.pi)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Magnitude plot (in dB)
    plt.subplot(2, 1, 1)
    plt.plot(freqs, 20 * np.log10(abs(h)))
    plt.title("Frequency Response")
    plt.ylabel("Magnitude [dB]")
    plt.xscale("log")
    plt.grid(True)

    # Phase plot (in degrees)
    plt.subplot(2, 1, 2)
    angles = np.unwrap(np.angle(h))
    plt.plot(freqs, np.degrees(angles))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [degrees]")
    plt.xscale("log")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("riaa_bode.pdf")
