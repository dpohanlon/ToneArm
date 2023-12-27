import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

from tqdm import tqdm

import argparse

import numpy as np

from scipy.io import wavfile

from scipy.signal import resample, savgol_filter

from Coil import Coil
from Stylus import Stylus
from Processing import (
    low_pass_filter,
    riaa_filter,
    normalize_audio,
    filter_stylus_radius,
    bump,
)


def write_to_wav(filename, audio, sample_rate=44100):
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


def noise_hiss(data, density=1000):

    length = len(data)
    number = int(length / density)

    for s in np.random.randint(0, length, size=number):

        l = 100
        a = np.clip(np.random.normal(1e-4, 1e-4), 0, 2e-4)
        f = np.random.randint(5000, 12000)

        data[s : s + l] += a * bump(freq=f, length=l)

    return data


def noise_pops(data, density=5000):

    length = len(data)
    number = int(length / density)

    for s in np.random.randint(0, length, size=number):

        l = np.random.choice([100, 200, 500, 1000], p=[0.45, 0.3, 0.2, 0.05])
        p = np.array([1, 1, 4, 4, 2, 1, 0.5])
        p /= np.sum(p)

        a = np.clip(np.random.normal(0.001, 0.001), 0, 0.002)

        data[s : s + l] += a * bump(
            freq=np.random.choice([10, 50, 100, 500, 1000, 2000, 5000], p=p), length=l
        )

    return data

def run_simulation(file_name, output_name, hiss_density, pop_density, total_time):

    stylus = Stylus()

    coil = Coil(
        coil_radius=1e-2,
        number_of_turns=1000,
        remanence=1.0,
        magnet_volume=0.01 * 0.01 * 0.01,
    )

    # Fudge factor so that maximum deviation causes slight distortion at ~5mV peak-peak

    ticks, data = wavfile.read(file_name)

    # If stero, make mono
    if len(data.shape) > 1:
        data = np.mean(data, axis = 1)

    data = data.astype("float")

    if total_time != None:
        data = data[: int(ticks * total_time)]

    data = normalize_audio(data)
    data = stylus.groove_pitch * data

    data = noise_hiss(data, density = hiss_density)
    data = noise_pops(data, density = pop_density)

    # Not working, for some reason
    # data = riaa_filter(data, mode = 'recording')

    voltages = []
    fluxes = []
    for i in tqdm(range(len(data) - 2)):
        voltage, dFlux = coil.induced_voltage(
            initial_distance=data[i],
            final_distance=data[i + 1],
            time_interval=1 / ticks,
        )
        voltages.append(voltage)
        fluxes.append(dFlux)

    voltages = np.array(voltages)
    voltages += np.random.normal(0, 1e-4, size=len(voltages))

    voltages = riaa_filter(voltages, mode="playback")

    norm_voltages = normalize_audio(voltages)

    voltages_filtered = filter_stylus_radius(norm_voltages, stylus)
    write_to_wav(f"{output_name}.wav", voltages_filtered)

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "--input",
        type=str,
        dest="file_name",
        default="test.wav",
        help="Input wave file.",
    )

    argParser.add_argument(
        "--name",
        type=str,
        dest="output_name",
        default="test",
        help="Output file name.",
    )

    argParser.add_argument(
        "--hiss", type=int, dest="hiss_density", default=1000, help="Hiss noise density."
    )

    argParser.add_argument(
        "--pop", type=int, dest="pop_density", default=5000, help="Pop noise density."
    )

    argParser.add_argument(
        "--length", type=float, dest="length", default=None, help="Max output size (seconds)."
    )

    args = argParser.parse_args()

    run_simulation(args.file_name, args.output_name, args.hiss_density, args.pop_density, args.length)
