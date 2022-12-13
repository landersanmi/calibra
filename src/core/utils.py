import logging
import numpy as np
import os

from jmetal.core.observer import Observer
from jmetal.core.solution import BinarySolution

from src.core.objectives import Objectives
from src.core.constraints import Constraints
from src.core.constants import (
    FITNESSES_FILENAME,
    PARETO_FILENAME,
    INFRASTRUCTURE_FILENAME,
    NETWORK_INFRASTRUCTURE_FILENAME,
    PIPELINE_FILENAME,
    LATENCIES_FILENAME
)

from src.core.models.pipeline import Pipeline
from src.core.models.infrastructure import Infrastructure
from src.core.models.network_infrastructure import NetworkInfrastructure
from src.core.models.latencies import Latency


LOGGER = logging.getLogger("jmetal")


class ParetoTools:
    def __init__(self, front):
        self.front = front

    def save(self):
        with open(PARETO_FILENAME, "w") as pareto_file:
            for i, solution in enumerate(self.front):
                for j in range(len(solution.objectives)):
                    pareto_file.write(f"{abs(solution.objectives[j])}")
                    pareto_file.write(",") if j != len(solution.objectives)-1 else pareto_file.write(f"\n")


class WriteObjectivesToFileObserver(Observer):
    def __init__(self) -> None:
        self.filename = FITNESSES_FILENAME
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        LOGGER.info(f"Evaluations: {evaluations}")

        solutions = np.array([s.objectives for s in kwargs["SOLUTIONS"]])
        objectives = []
        for i in range(solutions.shape[1]):
            objectives = np.append(objectives, solutions[np.argsort(solutions[:, i])][:5].mean(0)[i])

        with open(self.filename, "a") as out_file:
            for i in range(len(objectives)):
                out_file.write(f"{abs(objectives[i])}")
                out_file.write(",") if i != len(objectives)-1 else out_file.write(f"\n")

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



class Evaluate:
    def __init__(self, file_solution: str):
        self.file_solution = file_solution

        s = np.genfromtxt(file_solution, delimiter=",")
        number_of_models, number_of_devices = s.shape

        self.solution = BinarySolution(
            number_of_variables=number_of_models,
            number_of_objectives=5
        )
        for i in range(number_of_models):
            self.solution.variables[i] = [False for _ in range(number_of_devices)]

        for i in range(number_of_models):
            for j in range(number_of_devices):
                if s[i][j]:
                    self.solution.variables[i][j] = True

        with open(PIPELINE_FILENAME.format(pipeline=40), "r") as input_data_file:
            input_pipeline = input_data_file.read()
        self.pipe = Pipeline(input_pipeline).load()
        self.infra = Infrastructure(INFRASTRUCTURE_FILENAME).load()
        self.net_infra = NetworkInfrastructure(NETWORK_INFRASTRUCTURE_FILENAME).load()
        self.c = Constraints(self.solution, self.infra, self.pipe)

    def cost(self) -> float:
        cost = Objectives().get_consumption(self.pipe, self.infra, self.solution)
        return cost

    def model_performance(self) -> float:
        model_performance = Objectives().get_performance(self.pipe, self.infra, self.solution)
        return model_performance

    def resilience(self) -> float:
        resilience = Objectives().get_resilience(self.infra, self.solution)
        return resilience

    def network_performance(self) -> float:
        ld = Latency(LATENCIES_FILENAME).load()
        network_performance = Objectives().get_network_performance(ld, self.pipe, self.infra, self.solution)
        return network_performance

    def network_cost(self) -> float:
        network_cost = Objectives().get_net_cost(self.net_infra, self.solution)
        return network_cost

    def network_fail_probability(self) -> float:
        fail_probability = Objectives().get_net_fail_probability(self.net_infra, self.solution)
        return fail_probability

    def constraint_privacy(self) -> bool:
        return self.c.privacy_constraint()

    def constraint_cpu(self) -> bool:
        return self.c.cpu_constraint()

    def constraint_deployment(self) -> bool:
        return self.c.deployment_constraint()

    def constraint_ram(self) -> bool:
        return self.c.ram_constraint()

    def constraint_net_deployment(self) -> bool:
        return self.c.net_deployment_constraint()

    def constraint_net_device_capacity(self) -> bool:
        return self.c.net_device_capacity_constraint()

    def constraint_net_traffic_capacity(self) -> bool:
        return self.c.net_traffic_capacity_constraint()

    def constraint_net_layers(self) -> bool:
        return self.c.net_layers_constraint()