import numpy as np


class Stylus(object):
    def __init__(self, ticks=44100):

        self.ticks = ticks

        self.radius = 25e-6  # m
        self.groove_width = 0.05e-3  # m
        self.groove_pitch = 1e-3  # m

        self.distance_per_second = 0.36  # m

        self.ticks_per_metre = self.ticks / self.distance_per_second
