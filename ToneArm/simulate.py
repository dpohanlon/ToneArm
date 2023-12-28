import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

from tqdm import tqdm

import argparse

import numpy as np

from scipy.io import wavfile

from scipy.interpolate import interp1d

from ToneArm.Coil import Coil
from ToneArm.Stylus import Stylus
from ToneArm.Processing import (
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

        if s + l > length:
            continue

        data[s : s + l] += a * bump(freq=f, length=l)

    return data


def noise_pops(data, density=5000):

    length = len(data)
    number = int(length / density)

    for s in np.random.randint(0, length, size=number):

        l = np.random.choice([100, 200, 500, 1000], p=[0.45, 0.3, 0.2, 0.05])

        if s + l > length:
            continue

        p = np.array([1, 1, 4, 4, 2, 1, 0.5])
        p /= np.sum(p)

        a = np.clip(np.random.normal(0.001, 0.001), 0, 0.002)

        data[s : s + l] += a * bump(
            freq=np.random.choice([10, 50, 100, 500, 1000, 2000, 5000], p=p), length=l
        )

    return data


def simulate_wow(audio, rate=44100, depth=0.001, freq=0.5):

    t = np.arange(len(audio)) / rate

    # Create a slow LFO for wow
    lfo = np.sin(2 * np.pi * freq * t) * depth

    new_t = t + lfo
    interpolate = interp1d(new_t, audio, kind="linear", fill_value="extrapolate")

    return interpolate(t)


def simulate_flutter(audio, rate=44100, depth=0.0001, freq=10):

    t = np.arange(len(audio)) / rate

    lfo = np.sin(2 * np.pi * freq * t) * depth

    return audio * (1 + lfo)


def run_simulation(
    file_name,
    output_name,
    hiss_density,
    pop_density,
    wow_depth,
    flutter_depth,
    total_time,
):

    stylus = Stylus()

    coil = Coil(
        coil_radius=1e-2,
        number_of_turns=1000,
        remanence=1.0,
        magnet_volume=0.01 * 0.01 * 0.01,
    )

    if file_name != None:

        ticks, data = wavfile.read(file_name)

    else:

        if total_time == None:
            total_time = 10

        ticks = 44100
        data = np.zeros(int(ticks * total_time), dtype=float)

    # If stero, make mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    data = data.astype("float")

    if total_time != None:
        data = data[: int(ticks * total_time)]

    data = normalize_audio(data)
    data = stylus.groove_pitch * data

    wow_freq, flutter_freq = stylus.calculate_wow_flutter_frequencies()

    data = simulate_wow(data, freq=wow_freq, depth=wow_depth)
    data = simulate_flutter(data, freq=flutter_freq, depth=flutter_depth)

    data = noise_hiss(data, density=hiss_density)
    data = noise_pops(data, density=pop_density)

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


def run_tonearm():

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "--input",
        type=str,
        dest="file_name",
        default=None,
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
        "--hiss",
        type=int,
        dest="hiss_density",
        default=1000,
        help="Hiss noise density.",
    )

    argParser.add_argument(
        "--pop", type=int, dest="pop_density", default=5000, help="Pop noise density."
    )

    argParser.add_argument(
        "--wow",
        type=float,
        dest="wow_depth",
        default=0.0005,
        help="Wow (low frequency distortion) depth.",
    )

    argParser.add_argument(
        "--flutter",
        type=float,
        dest="flutter_depth",
        default=0.0001,
        help="flutter (high frequency distortion) depth.",
    )

    argParser.add_argument(
        "--length",
        type=float,
        dest="length",
        default=None,
        help="Max output size (seconds).",
    )

    args = argParser.parse_args()

    run_simulation(
        args.file_name,
        args.output_name,
        args.hiss_density,
        args.pop_density,
        args.wow_depth,
        args.flutter_depth,
        args.length,
    )


if __name__ == "__main__":

    run_tonearm()
