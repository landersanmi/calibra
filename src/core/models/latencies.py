import numpy as np


class Latency:
    def __init__(self, file_location: str):
        ld = np.loadtxt(file_location, dtype=float)
        x, y = ld.shape
        # normalize
        ld /= ld.max() * (x * y)
        self.ld = np.reshape(ld, (x, 1, y))

    def load(self):
        return self.ld
