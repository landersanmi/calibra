import numpy as np
import logging
import datetime
import pandas as pd

from src.core.constants import CONSTRAINT_LABELS

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
    def __init__(self, logger):
        super(StoppingByConstraintsMet, self).__init__()
        self.constraints_met = False
        self.generations = 0
        self.tensorboard_logger = logger

    def update(self, *args, **kwargs):
        seconds = kwargs["COMPUTING_TIME"]
        #evaluations = kwargs["EVALUATIONS"]
        c = np.array([s.constraints for s in kwargs["SOLUTIONS"]])
        df = pd.DataFrame(c, columns=CONSTRAINT_LABELS)

        uncompleted_constraint = False
        constraints_sizes = dict()
        constraints_sizes['size total'] = len(df)
        for constraint in CONSTRAINT_LABELS:
            constraints_sizes['size ' + constraint] = sum(df[constraint] == 0)

            if df[constraint].any() != 0:
                uncompleted_constraint = True

        if not uncompleted_constraint:
            self.constraints_met = True

        log_msg = str(datetime.timedelta(seconds=seconds)) + " || "
        for size in constraints_sizes:
            log_msg += (size + '=' + str(constraints_sizes[size]) + ' | ')
        print(log_msg)
        LOGGER.info(log_msg)

        del constraints_sizes['size total']
        self.tensorboard_logger.log_constraints(constraints=constraints_sizes, constraints_met=self.constraints_met, x_axis_value=self.generations)
        self.generations += 1

    @property
    def is_met(self):
        return self.constraints_met


class StoppingByGenerationsAfterConstraintsMet(TerminationCriterion):
    def __init__(self, generations, logger):
        super(StoppingByGenerationsAfterConstraintsMet, self).__init__()
        self.constraints_met = False
        self.generations_met = False
        self.generations = generations
        self.generations_after_constraints = 0
        self.all_generations = 0
        self.tensorboard_logger = logger

    def update(self, *args, **kwargs):
        seconds = kwargs["COMPUTING_TIME"]
        #evaluations = kwargs["EVALUATIONS"]
        c = np.array([s.constraints for s in kwargs["SOLUTIONS"]])
        df = pd.DataFrame(c, columns=CONSTRAINT_LABELS)

        uncompleted_constraint = False
        constraints_sizes = dict()
        constraints_sizes['size total'] = len(df)
        for constraint in CONSTRAINT_LABELS:
            constraints_sizes['size '+constraint] = sum(df[constraint] == 0)

            if df[constraint].any() != 0:
                uncompleted_constraint = True

        if not uncompleted_constraint:
            self.constraints_met = True

        if self.constraints_met:
            self.generations_after_constraints += 1
            self.generations_met = (self.generations_after_constraints == self.generations)
        else:
            log_msg = str(datetime.timedelta(seconds=seconds)) + " || "
            for size in constraints_sizes:
                log_msg += (size + '=' + str(constraints_sizes[size]) + ' | ')
            print(log_msg)
            LOGGER.info(log_msg)

        del constraints_sizes['size total']
        self.tensorboard_logger.log_constraints(constraints=constraints_sizes, constraints_met=self.constraints_met, x_axis_value=self.all_generations)
        self.all_generations += 1

    @property
    def is_met(self):
        return self.constraints_met and self.generations_met


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
