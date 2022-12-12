import pandas as pd


class NetworkInfrastructure:
    def __init__(self, file_location: str):
        self.infrastructure = pd.read_csv(file_location, sep=";")

    def load(self):
        return self.infrastructure