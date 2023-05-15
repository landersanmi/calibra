import pandas as pd
import numpy as np
import io


class Infrastructure:
    def __init__(self, input_data: str):
        bytes_io = io.BytesIO(input_data)
        self.infrastructure = pd.read_csv(bytes_io)

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
