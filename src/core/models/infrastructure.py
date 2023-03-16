import pandas as pd
import numpy as np


class Infrastructure:
    def __init__(self, file_location: str):
        self.infrastructure = pd.read_csv(file_location)

        x = lambda txt: np.fromstring(txt[1:-1], sep=",")
        self.infrastructure.consumption = self.infrastructure.consumption.apply(x)
        self.infrastructure.parallelization = self.infrastructure.parallelization.apply(x)

        # normalize consumption
        consumption_max = self.infrastructure.consumption.apply(max).max()
        consumption_min = self.infrastructure.consumption.apply(min).min()
        self.infrastructure.consumption = (self.infrastructure.consumption - consumption_min) / (
            consumption_max - consumption_min
        )

        # normalize performance
        performance_max = self.infrastructure.performance.max()
        performance_min = self.infrastructure.performance.min()
        self.infrastructure.performance = (self.infrastructure.performance - performance_min) / (
            performance_max - performance_min
        )

    def load(self):
        return self.infrastructure
