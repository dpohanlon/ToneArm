import numpy as np


class Stylus(object):
    def __init__(self, ticks=44100):

        self.ticks = ticks

        self.radius = 25e-6  # m
        self.groove_width = 0.05e-3  # m
        self.groove_pitch = 1e-3  # m

        self.record_speed_rpm = 33.333

        self.record_speed_hz = self.record_speed_rpm / 60

        self.distance_per_second = 0.36  # m

        self.ticks_per_metre = self.ticks / self.distance_per_second

    def calculate_wow_flutter_frequencies(self, wow=2, flutter=20):

        record_speed_hz = self.record_speed_rpm / 60

        wow_freq = record_speed_hz * wow

        flutter_freq = record_speed_hz * flutter

        return wow_freq, flutter_freq
