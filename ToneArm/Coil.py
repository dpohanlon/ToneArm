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

    distance_fudge_calib()
