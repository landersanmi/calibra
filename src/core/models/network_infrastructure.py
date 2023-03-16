import pandas as pd


class NetworkInfrastructure:
    def __init__(self, file_location: str):
        self.net_infrastructure = pd.read_csv(file_location, sep=";")

        # normalize cost
        cost_max = self.net_infrastructure.cost.max()
        cost_min = self.net_infrastructure.cost.min()
        self.net_infrastructure.cost = (self.net_infrastructure.cost - cost_min) / (
            cost_max - cost_min
        )

        # normalize fail probability
        fail_prob_max = self.net_infrastructure.failure_prob.max()
        fail_prob_min = self.net_infrastructure.failure_prob.min()
        self.net_infrastructure.failure_prob = (self.net_infrastructure.failure_prob - fail_prob_min) / (
            fail_prob_max - fail_prob_min
        )

    def load(self):
        return self.net_infrastructure
