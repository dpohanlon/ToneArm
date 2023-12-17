import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

from tqdm import tqdm

import numpy as np

from scipy.io import wavfile

class Coil:
    def __init__(self, coil_radius, number_of_turns, remanence, magnet_volume, distance_fudge = 0.01, flux_fudge = 1E-4):
        """
        Initializes the Coil object with given parameters.

        Parameters:
        coil_radius (float): Radius of the coil in meters.
        number_of_turns (int): Number of turns in the coil.
        remanence (float): Remanence of the neodymium magnet (in Tesla).
        magnet_volume (float): Volume of the neodymium magnet (in cubic meters).
        """

        self.coil_radius = coil_radius
        self.number_of_turns = number_of_turns
        self.remanence = remanence
        self.magnet_volume = magnet_volume
        self.resting_distance = 1E-4

        self.distance_fudge = distance_fudge # 0.01, Slight distortion at maximum deflection

        self.flux_fudge = flux_fudge # 5mV at maximum deflection

        self.mu0 = 4 * np.pi * 10**-7  # Permeability of free space

    def calculate_flux(self, distance):
        """
        Calculates the magnetic flux through the coil at a given distance from the magnet.

        Parameters:
        distance (float): Distance from the magnet to the coil (in meters).

        Returns:
        float: Magnetic flux through the coil (in Weber, Wb).
        """

        # Area of the coil
        area = np.pi * self.coil_radius**2

        # Magnetic field strength at the given distance
        B = (self.mu0 / (4 * np.pi)) * ((2 * self.remanence * self.magnet_volume) / (self.resting_distance + distance) ** 3)

        # Magnetic flux
        flux = B * area

        return flux

    def induced_voltage(self, initial_distance, final_distance, time_interval):
        """
        Calculates the induced voltage in the coil due to the movement of the magnet.

        Parameters:
        initial_distance (float): Initial distance from the magnet to the coil (in meters).
        final_distance (float): Final distance from the magnet to the coil (in meters).
        time_interval (float): Time interval over which the distance changes (in seconds).

        Returns:
        float: Induced voltage in the coil (in Volts).
        """

        initial_distance = self.distance_fudge * initial_distance
        final_distance = self.distance_fudge * final_distance

        # Change in magnetic flux
        initial_flux = self.calculate_flux(initial_distance)
        final_flux = self.calculate_flux(final_distance)
        flux_change = final_flux - initial_flux

        flux_change = flux_change * self.flux_fudge

        # Induced voltage (Faraday's Law of Induction)
        voltage = -self.number_of_turns * (flux_change / time_interval)

        return voltage, flux_change

class VinylCartridgeModel:
    def __init__(self, coil_width, magnet_strength, frequency, damping_factor=0.1, sampling_rate=10000):
        self.coil_width = coil_width  # Width of the coil
        self.magnet_strength = magnet_strength  # Strength of the magnetic field
        self.frequency = frequency  # Frequency of the audio signal (Hz)
        self.damping_factor = damping_factor  # Damping factor to smooth out spikes
        self.sampling_rate = sampling_rate  # Sampling rate for the simulation

    def simulate(self, duration):
        time = np.linspace(0, duration, int(self.sampling_rate * duration))
        magnet_position = 0.1 * np.sin(2 * np.pi * self.frequency * time)  # Simple harmonic motion

        # Simulate the change in magnetic flux and induced voltage for each small section of the coil
        num_sections = 1000  # Divide the coil into 100 small sections
        section_width = self.coil_width / num_sections
        voltages = np.zeros_like(time)

        for section in range(num_sections):
            section_center = (section + 0.5) * section_width
            distance_from_magnet = np.abs(section_center - magnet_position)
            # Modified flux change calculation with damping factor
            flux_change = self.magnet_strength / (distance_from_magnet + self.damping_factor)
            voltages += np.gradient(flux_change, time)  # Faraday's Law of Induction

        return time, voltages

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

def distance_fudge_calib():

    groove_width = 0.05E-3
    groove_pitch = 1E-3

    # 1 second, 0.36m

    freq = 440 # Hz

    coil = Coil(coil_radius=1E-2, number_of_turns=1000, remanence=1.0, magnet_volume= 0.01 * 0.01 * 0.01, distance_fudge = 1)

    ticks = 44100
    total_time = 1

    # Fudge factor so that maximum deviation causes slight distortion at ~5mV peak-peak

    voltages_fudge = []
    fudges = [0.001, 0.005, 0.01, 0.02, 0.03]

    for fudge in fudges:

        data = fudge * groove_pitch * np.sin(freq * 2 * np.pi * np.linspace(0, total_time, ticks))

        deltaPos = data[1:] - data[:-1]

        voltages = []
        fluxes = []
        for i in tqdm(range(len(data) - 2)):
            voltage, dFlux = coil.induced_voltage(initial_distance=data[i], final_distance=data[i + 1], time_interval=total_time/ticks)
            voltages.append(voltage)
            fluxes.append(dFlux)

        voltages = np.array(voltages)

        voltages_fudge.append(voltages)

    for i, fudge in enumerate(fudges):

        v = voltages_fudge[i][:200]
        v /= np.sum(v)

        plt.plot(np.linspace(0, total_time, ticks)[:-2][:200], v, label = str(fudge))

    plt.legend(loc = 0)
    plt.savefig('fudges.pdf')
    plt.clf()

if __name__ == '__main__':

    groove_width = 0.05E-3
    groove_pitch = 1E-3

    # 1 second, 0.36m

    freq = 440 # Hz

    coil = Coil(coil_radius=1E-2, number_of_turns=1000, remanence=1.0, magnet_volume= 0.01 * 0.01 * 0.01)

    ticks = 44100
    total_time = 1

    # Fudge factor so that maximum deviation causes slight distortion at ~5mV peak-peak

    data = groove_pitch * np.sin(freq * 2 * np.pi * np.linspace(0, total_time, ticks))
    data += np.random.normal(0, 0.005 * np.mean(np.abs(data)), size = data.shape)

    plt.plot(np.linspace(0, total_time, ticks)[:250], data[:250])
    plt.savefig('a.pdf')
    plt.clf()

    deltaPos = data[1:] - data[:-1]

    voltages = []
    fluxes = []
    for i in tqdm(range(len(data) - 2)):
        voltage, dFlux = coil.induced_voltage(initial_distance=data[i], final_distance=data[i + 1], time_interval=total_time/ticks)
        voltages.append(voltage)
        fluxes.append(dFlux)

    plt.plot(fluxes[:1000])
    plt.savefig('f.pdf')
    plt.clf()

    voltages = np.array(voltages)

    print(deltaPos[:-1][voltages < 100][:100])

    plt.plot(np.linspace(0, total_time, ticks)[:-2][:100], voltages[:100])

    plt.savefig('v.pdf')
    plt.clf()

    norm_voltages = normalize_audio(voltages)
    write_to_wav('test.wav', norm_voltages)
