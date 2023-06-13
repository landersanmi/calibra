import numpy as np
import logging
import datetime
import pandas as pd

from swagger_server.ai.constants import CONSTRAINT_LABELS

from jmetal.util.termination_criterion import TerminationCriterion

LOGGER = logging.getLogger("jmetal")


class StoppingByConstraintsMet(TerminationCriterion):
    def __init__(self, logger=None):
        super(StoppingByConstraintsMet, self).__init__()
        self.constraints_met = False
        self.generations = 0
        self.tensorboard_logger = logger
        self.total_seconds = 0

    def update(self, *args, **kwargs):
        self.total_seconds = kwargs["COMPUTING_TIME"]
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

        log_msg = str(datetime.timedelta(seconds=self.total_seconds)) + " || "
        for size in constraints_sizes:
            log_msg += (size + '=' + str(constraints_sizes[size]) + ' | ')
        LOGGER.info(log_msg)

        del constraints_sizes['size total']
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_constraints(constraints=constraints_sizes, constraints_met=self.constraints_met, x_axis_value=self.generations)
        self.generations += 1

    @property
    def is_met(self):
        return self.constraints_met


class StoppingByGenerationsAfterConstraintsMet(TerminationCriterion):
    def __init__(self, max_generations, logger=None):
        super(StoppingByGenerationsAfterConstraintsMet, self).__init__()
        self.constraints_met = False
        self.generations_met = False
        self.generations = max_generations
        self.generations_after_constraints = 0
        self.all_generations = 0
        self.tensorboard_logger = logger
        self.total_seconds = 0
        self.seconds_to_met_constraints = 0

    def update(self, *args, **kwargs):
        self.total_seconds = kwargs["COMPUTING_TIME"]
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
            self.generations_after_constraints += 1
            self.generations_met = (self.generations_after_constraints == self.generations)
        else:
            log_msg = str(datetime.timedelta(seconds=self.total_seconds)) + " || "
            for size in constraints_sizes:
                log_msg += (size + '=' + str(constraints_sizes[size]) + ' | ')
            LOGGER.info(log_msg)

        del constraints_sizes['size total']
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_constraints(constraints=constraints_sizes, constraints_met=self.constraints_met, x_axis_value=self.all_generations)
        self.all_generations += 1

    @property
    def is_met(self):
        return self.constraints_met and self.generations_met


class StoppingByTimeAfterConstraintsMet(TerminationCriterion):
    def __init__(self, max_seconds, logger=None):
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
            LOGGER.info(log_msg)

        del constraints_sizes['size total']
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_constraints(constraints=constraints_sizes, constraints_met=self.constraints_met, x_axis_value=self.generations)
        self.generations += 1

    @property
    def is_met(self):
        return self.constraints_met and self.time_met


class StoppingByTimeOrGenerationsAfterConstraintsMet(TerminationCriterion):
    def __init__(self, max_seconds, max_generations, logger=None):
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
            LOGGER.info(log_msg)

        del constraints_sizes['size total']
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_constraints(constraints=constraints_sizes, constraints_met=self.constraints_met, x_axis_value=self.all_generations)
        self.all_generations += 1

    @property
    def is_met(self):
        return self.constraints_met and (self.time_met or self.generations_met)
