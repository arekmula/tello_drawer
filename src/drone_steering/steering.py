import numpy as np


class DroneSteering:
    def __init__(self, max_area=100, max_speed=10, signal_period=1):
        self.max_area = max_area
        self.max_speed = max_speed
        self.signal_period = signal_period