import logging
import numpy as np
import json

from jmetal.core.observer import Observer
from jmetal.core.solution import CompositeSolution
LOGGER = logging.getLogger("jmetal")


class WriteObjectivesToTensorboardObserver(Observer):
    def __init__(self, logger) -> None:
        self.generations = 0
        self.tensorboard_logger = logger

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        LOGGER.info(f"Evaluations: {evaluations}")

        solutions = np.array([s.objectives for s in kwargs["SOLUTIONS"]])
        objectives = []
        for i in range(solutions.shape[1]):
            objectives = np.append(objectives, solutions[np.argsort(solutions[:, i])][:5].mean(0)[i])

        objectives = abs(objectives)
        self.tensorboard_logger.log_objectives(objectives, self.generations)
        self.generations += 1


# Custom JSON encoder class
class CompositeSolutionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, CompositeSolution):
            return {
                'variables_0': obj.variables[0].variables.tolist(),
                'variables_1': obj.variables[1].variables.tolist(),
                'objectives': self._handle_serialization(obj.objectives),
                'constraints': self._handle_serialization(obj.constraints)
            }
        return super().default(obj)

    def _handle_serialization(self, data):
        if isinstance(data, list):
            return [self._handle_serialization(item) for item in data]
        if isinstance(data, (np.ndarray, np.number)):
            return data.tolist()
        return data