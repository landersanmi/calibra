from src.core.constants import CONTINENT_CODES
import pycountry
import pycountry_convert as pc
import pandas as pd
import numpy as np


class Infrastructure:
    def __init__(self, file_location: str):
        self.infrastructure = pd.read_csv(file_location)

        x = lambda txt: np.fromstring(txt[1:-1], sep=",")
        self.infrastructure.consumption = self.infrastructure.consumption.apply(x)

        # normalize consumption
        my_max = self.infrastructure.consumption.apply(max).max()
        my_min = self.infrastructure.consumption.apply(min).min()
        self.infrastructure.consumption = (self.infrastructure.consumption - my_min) / (
            my_max - my_min
        )
        # self.infrastructure.consumption /= ceil(self.infrastructure.consumption.apply(max).sum())

        self.infrastructure.parallelization = self.infrastructure.parallelization.apply(x)

        y = lambda row: int(pycountry.countries.get(alpha_2=row["country_code"]).numeric)
        self.infrastructure["country"] = self.infrastructure.apply(y, axis=1)

        # convert alpha2 country_code to continent name
        z = lambda row: CONTINENT_CODES[
            pc.country_alpha2_to_continent_code(row["country_code"])
        ]
        self.infrastructure["continent"] = self.infrastructure.apply(z, axis=1)

        # normalize bandwidth
        self.infrastructure.bandwidth = (
            self.infrastructure.bandwidth / self.infrastructure.bandwidth.max()
        )

        # normalize performance
        self.infrastructure.performance = (
            self.infrastructure.performance - self.infrastructure.performance.min()
        ) / (
            self.infrastructure.performance.max()
            - self.infrastructure.performance.min()
        )

        # normalize resilience
        self.infrastructure.resillience = (
            self.infrastructure.resillience - self.infrastructure.resillience.min()
        ) / (
            self.infrastructure.resillience.max()
            - self.infrastructure.resillience.min()
        )

    def load(self):
        return self.infrastructure