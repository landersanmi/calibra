import logging
import numpy as np
import random

from src.core.objectives import Objectives
from src.core.constraints import Constraints

from src.core.models.pipeline import Pipeline
from src.core.models.infrastructure import Infrastructure
from src.core.models.network_infrastructure import NetworkInfrastructure

from src.core.constants import OBJECTIVES_LABELS, CONSTRAINT_LABELS

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import CompositeSolution, BinarySolution


class DeploymentProblem(BinaryProblem):
    def __init__(self, file_infrastructure, file_network_infrastructure, input_pipeline):
        super(DeploymentProblem, self).__init__()

        self.infra = Infrastructure(file_infrastructure).load()
        self.net_infra = NetworkInfrastructure(file_network_infrastructure).load()
        self.pipe = Pipeline(input_pipeline).load()

        self.objectives = None
        self.obj_labels = OBJECTIVES_LABELS

        self.number_of_models = self.pipe.shape[0]
        self.number_of_objectives = len(self.obj_labels)
        self.number_of_devices = len(self.infra.index)
        self.number_of_net_devices = len(self.net_infra.index)
        self.number_of_constraints = len(CONSTRAINT_LABELS)

    def evaluate(self, solution: CompositeSolution) -> CompositeSolution:
        self.objectives = Objectives()

        solution.objectives[0] = -1 * self.objectives.get_performance(
            self.pipe, self.infra, solution
        )
        solution.objectives[1] = self.objectives.get_consumption(
            self.pipe, self.infra, solution
        )
        solution.objectives[2] = self.objectives.get_net_cost(
            net_infra=self.net_infra, solution=solution
        )
        solution.objectives[3] = self.objectives.get_net_fail_probability(
            net_infra=self.net_infra, solution=solution
        )

        self.__evaluate_constraints(solution)
        return solution

    def __evaluate_constraints(self, solution: CompositeSolution) -> None:
        constraints = []
        c = Constraints(solution, self.infra, self.net_infra, self.pipe)

        """ 
        do not exceed total CPU per device
        """
        constraints.append(c.cpu_constraint())

        """
        do not exceed total RAM per device
        """
        constraints.append(c.ram_constraint())

        """ 
        each model should be deployed in at least one device
        """
        constraints.append(c.deployment_constraint())

        """
        the bandwidth of each device must not be exceeded by the 
        sum of models network requirements deployed on it
        """
        constraints.append(c.bandwidth_constraint())
        """
        each device with at least one model deployed 
        should be connected to one network device.
        """
        constraints.append(c.net_deployment_constraint())

        """
        each net device maximum users capacity must be complained
        """
        constraints.append(c.net_device_capacity_constraint())

        """
        each net device maximum traffic capacity must be complained
        """
        constraints.append(c.net_traffic_capacity_constraint())

        """
        each device must be related to a net device of the same layer
        """
        constraints.append(c.net_layers_constraint())

        solution.constraints = constraints

    def create_solution(self) -> CompositeSolution:
        model_solution = BinarySolution(number_of_objectives=self.number_of_objectives,
                                        number_of_constraints=self.number_of_constraints,
                                        number_of_variables=self.number_of_models)
        network_solution = BinarySolution(number_of_objectives=self.number_of_objectives,
                                          number_of_constraints=self.number_of_constraints,
                                          number_of_variables=self.number_of_net_devices)

        for i in range(self.number_of_models):
            model_solution.variables[i] = [
               True if random.random() > 0.5 else False
               for _ in range(self.number_of_devices)
            ]

        net_sol = list()
        for i in range(self.number_of_devices):
            rand_net = random.randint(0, self.number_of_net_devices)
            net_sol.append([
               True if j == rand_net else False for j in range(self.number_of_net_devices)
            ])
        net_sol = np.array(net_sol)
        network_solution.variables = net_sol.transpose()
        new_solution = CompositeSolution([model_solution, network_solution])
        return new_solution

    def get_name(self) -> str:
        return "DeploymentProblem"

