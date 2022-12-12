import numpy as np
import logging
import datetime
import pandas as pd

from jmetal.util.comparator import DominanceComparator
from jmetal.util.termination_criterion import TerminationCriterion

LOGGER = logging.getLogger("jmetal")


class StoppingByNonDominance(TerminationCriterion):
    def __init__(self, idle_evaluations: int):
        super(StoppingByNonDominance, self).__init__()
        self.idle_evaluations = idle_evaluations
        self.evaluations = 0
        self.best_solution = None

    def update(self, *args, **kwargs):
        current_solution = kwargs["SOLUTIONS"][0]
        if self.best_solution is None:
            self.best_solution = current_solution
        else:
            result = DominanceComparator().compare(self.best_solution, current_solution)
            if result != 1:
                self.evaluations += 1
                LOGGER.info(f"{self.evaluations} evaluations without improvement.")
            else:
                self.evaluations = 0
            self.best_solution = current_solution

    @property
    def is_met(self):
        return self.evaluations >= self.idle_evaluations


class StoppingByTotalDominance(TerminationCriterion):
    def __init__(self, idle_evaluations: int):
        super(StoppingByTotalDominance, self).__init__()
        self.idle_evaluations = idle_evaluations
        self.evaluations = 0
        self.best_objectives = None
        self.seconds = 0.0

    def update(self, *args, **kwargs):
        self.seconds = kwargs["COMPUTING_TIME"]

        s = np.array([s.objectives for s in kwargs["SOLUTIONS"]])
        objective0 = s[np.argsort(s[:, 0])][0][0]
        objective1 = s[np.argsort(s[:, 1])][0][1]
        objective2 = s[np.argsort(s[:, 2])][0][2]
        objective3 = s[np.argsort(s[:, 3])][0][3]
        objective4 = s[np.argsort(s[:, 4])][0][4]
        objective5 = s[np.argsort(s[:, 5])][0][5]
        current_objectives = [objective0, objective1, objective2, objective3, objective4, objective5]

        s = kwargs["SOLUTIONS"][0]
        if self.best_objectives is None:
            self.best_objectives = current_objectives
            # LOGGER.info(self.best_objectives)
        else:
            # LOGGER.info(self.best_objectives)
            # LOGGER.info(current_objectives)
            if all([x <= y for x, y in zip(self.best_objectives, current_objectives)]):
                self.evaluations += 1
            else:
                self.best_objectives = [
                    min(x, y) for x, y in zip(self.best_objectives, current_objectives)
                ]
                self.evaluations = 0
        LOGGER.info(f"{self.evaluations} evaluations without improvement")

    @property
    def is_met(self):
        return self.evaluations >= (self.idle_evaluations - (self.seconds / 14) ** 2)
        # return self.evaluations >= (self.idle_evaluations)


class StoppingByConstraintsMet(TerminationCriterion):
    def __init__(self):
        super(StoppingByConstraintsMet, self).__init__()
        self.constraints_met = False

    def update(self, *args, **kwargs):
        seconds = kwargs["COMPUTING_TIME"]

        c = np.array([s.constraints for s in kwargs["SOLUTIONS"]])
        df = pd.DataFrame(
            c,
            columns=[
                "cpu",
                "ram",
                "deploy",
                #"privacy",
                "net deployment",
                "net device capacity",
                "net traffic capacity",
                "net layers"
            ],
        )
        size = sum(
            (
                (df["cpu"] == 0)
                & (df["ram"] == 0)
                & (df["deploy"] == 0)
                #& (df["privacy"] == 0)
                & (df["net deployment"] == 0)
                & (df["net device capacity"] == 0)
                & (df["net traffic capacity"] == 0)
                & (df["net layers"] == 0)
            )
        )
        if size > 0:
            self.constraints_met = True

        size_cpu = sum(df["cpu"] == 0)
        size_ram = sum(df["ram"] == 0)
        size_deploy = sum(df["deploy"] == 0)
        #size_privacy = sum(df["privacy"] == 0)
        size_net_deploy = sum(df["net deployment"] == 0)
        size_net_dev_capacity = sum(df["net device capacity"] == 0)
        size_net_traffic_capacity = sum(df["net traffic capacity"] == 0)
        size_net_layers = sum(df["net layers"] == 0)
        size_total = len(df)

        LOGGER.info(
            f"{str(datetime.timedelta(seconds=seconds))} - total = {size_total}, cpu = {size_cpu}, ram = {size_ram}, deploy = {size_deploy}, net deploy = {size_net_deploy}, net dev capacity = {size_net_dev_capacity}, net traffic capacity = {size_net_traffic_capacity}, net layers = {size_net_layers}"
        )

    @property
    def is_met(self):
        return self.constraints_met


class StoppingByFullPareto(TerminationCriterion):
    def __init__(self, offspring_size: int):
        super(StoppingByFullPareto, self).__init__()
        self.offspring_size = offspring_size

    def update(self, *args, **kwargs):
        self.current_offspring_size = len(kwargs["SOLUTIONS"])
        LOGGER.info(f"Current size = {self.current_offspring_size}.")

    @property
    def is_met(self):
        return self.offspring_size <= self.current_offspring_size

