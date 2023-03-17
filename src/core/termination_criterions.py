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
        current_objectives = [objective0, objective1, objective2, objective3]

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


class StoppingByTimeAfterConstraintsMet(TerminationCriterion):
    def __init__(self, max_seconds, logger):
        super(StoppingByTimeAfterConstraintsMet, self).__init__()
        self.constraints_met = False
        self.time_met = False
        self.max_seconds = max_seconds
        self.total_seconds = 0
        self.seconds_to_met_constraints = 0
        self.generations = 0
        self.tensorboard_logger = logger

    def update(self, *args, **kwargs):
        self.total_seconds = kwargs["COMPUTING_TIME"]
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
            if self.seconds_to_met_constraints == 0:
                self.seconds_to_met_constraints = self.total_seconds
            self.time_met = (self.max_seconds <= (self.total_seconds - self.seconds_to_met_constraints))
            if self.time_met:
                print("Optimization time: {} seconds".format(self.total_seconds - self.seconds_to_met_constraints))
        else:
            log_msg = str(datetime.timedelta(seconds=self.total_seconds)) + " || "
            for size in constraints_sizes:
                log_msg += (size + '=' + str(constraints_sizes[size]) + ' | ')
            print(log_msg)
            LOGGER.info(log_msg)

        del constraints_sizes['size total']
        self.tensorboard_logger.log_constraints(constraints=constraints_sizes, constraints_met=self.constraints_met, x_axis_value=self.generations)
        self.generations += 1

    @property
    def is_met(self):
        return self.constraints_met and self.time_met


class StoppingByTimeOrGenerationsAfterConstraintsMet(TerminationCriterion):
    def __init__(self, max_seconds, max_generations, logger):
        super(StoppingByTimeOrGenerationsAfterConstraintsMet, self).__init__()
        self.constraints_met = False

        self.time_met = False
        self.max_seconds = max_seconds
        self.total_seconds = 0
        self.seconds_to_met_constraints = 0

        self.generations_met = False
        self.max_generations = max_generations
        self.generations_after_constraints = 0
        self.all_generations = 0

        self.tensorboard_logger = logger

    def update(self, *args, **kwargs):
        self.total_seconds = kwargs["COMPUTING_TIME"]
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
            if self.seconds_to_met_constraints == 0:
                self.seconds_to_met_constraints = self.total_seconds
            self.time_met = (self.max_seconds <= (self.total_seconds - self.seconds_to_met_constraints))

            self.generations_after_constraints += 1
            self.generations_met = (self.generations_after_constraints == self.max_generations)

            if self.time_met:
                print("Optimization finished by time: {} seconds".format(self.total_seconds - self.seconds_to_met_constraints))
            if self.generations_met:
                print("Optimization finished by generations: {} generations".format(self.generations_after_constraints))

        else:
            log_msg = str(datetime.timedelta(seconds=self.total_seconds)) + " || "
            for size in constraints_sizes:
                log_msg += (size + '=' + str(constraints_sizes[size]) + ' | ')
            print(log_msg)
            LOGGER.info(log_msg)

        del constraints_sizes['size total']
        self.tensorboard_logger.log_constraints(constraints=constraints_sizes, constraints_met=self.constraints_met, x_axis_value=self.all_generations)
        self.all_generations += 1

    @property
    def is_met(self):
        return self.constraints_met and (self.time_met or self.generations_met)


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

