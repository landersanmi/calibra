import unittest
# from unittest.mock import patch

import logging
import numpy as np

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import CompositeSolution, BinarySolution
from jmetal.util.termination_criterion import StoppingByEvaluations
from src.core.problem import TravelingModel

from src.core.optimizer import Optimizer
from src.core.constraints import Constraints

from src.core.models.infrastructure import Infrastructure
from src.core.models.pipeline import Pipeline

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning)


class TestConstraints(unittest.TestCase):
    def setUp(self):
        file_infrastructure = "tests/resources/infrastructure.csv"
        file_network_infrastructure = "tests/resources/network_infrastructure.csv"
        file_pipeline = "tests/resources/pipeline.yaml"
        with open(file_pipeline, "r") as input_data_file:
            input_pipeline = input_data_file.read()

        self.o = Optimizer(
            termination_criterion=StoppingByEvaluations(max_evaluations=1000),
            file_infrastructure=file_infrastructure,
            file_net_infrastructure=file_network_infrastructure,
            input_pipeline=input_pipeline,
            population_size=200,
            observer=None,
        )
        # observer=WriteObjectivesToFileObserver())
        self.o.run()
        self.front = self.o.get_front()
        # load pipeline
        self.pipe = Pipeline(input_pipeline).load()
        # load infrastructure
        self.infra = Infrastructure(file_infrastructure).load()

    # each model should be deployed in at least one device
    def test_deployment(self):
        for sol in self.front:
            print(sol.variables)
            s = np.asfarray(sol.variables[0].variables, dtype=np.bool)
            z = np.sum(s, axis=1) - 1
            self.assertFalse((z < 0).any())

            c = Constraints(sol, self.infra, self.pipe)
            self.assertEqual(c.deployment_constraint(), 0)

    # do not exceed total CPU per device
    def test_cpu(self):
        for sol in self.front:
            s = np.asfarray(sol.variables, dtype=np.bool)
            # i = s.transpose()*self.p[0]
            i = s.transpose() * self.pipe.cpus.to_numpy()
            j = np.sum(i, axis=1)
            k = self.infra.thread_count.to_numpy()
            self.assertFalse((k < j).any())

            c = Constraints(sol, self.infra, self.pipe)
            self.assertEqual(c.cpu_constraint(), 0)

    # do not exceed total memory per device
    def test_memory(self):
        for sol in self.front:
            s = np.asfarray(sol.variables, dtype=np.bool)
            i = s.transpose() * self.pipe.memory.to_numpy()
            j = np.sum(i, axis=1)
            k = self.infra.memory.to_numpy()
            self.assertFalse((k < j).any())

            c = Constraints(sol, self.infra, self.pipe)
            self.assertEqual(c.ram_constraint(), 0)

    def test_privacy(self):
        # for sol in self.front:
        c = Constraints(self.front[0], self.infra, self.pipe)
        self.assertTrue(c.privacy_constraint() >= 0)


if __name__ == "__main__":
    unittest.main()
