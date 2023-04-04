from tests.unit.core.constants import (
    OBJECTIVES_LABELS,
    CONSTRAINT_LABELS,
    TEST_NETWORK_INFRASTRUCTURE_FILENAME,
    TEST_PIPELINE_FILENAME,
    TEST_INFRASTRUCTURE_FILENAME,
    UNCONSTRAINED_INFRA_FILENAME,
    UNCONSTRAINED_NET_INFRA_FILENAME
)

import unittest
import numpy as np

from jmetal.core.solution import BinarySolution, CompositeSolution

from src.core.models.infrastructure import Infrastructure
from src.core.models.pipeline import Pipeline
from src.core.models.network_infrastructure import NetworkInfrastructure
from src.core.problem import DeploymentProblem


class TestProblem(unittest.TestCase):

    def setUp(self):
        self.file_pipeline = TEST_PIPELINE_FILENAME.format(pipeline="5NET")
        with open(self.file_pipeline, "r") as input_data_file:
            self.input_pipeline = input_data_file.read()

        self.problem = DeploymentProblem(file_infrastructure=TEST_INFRASTRUCTURE_FILENAME,
                                         file_network_infrastructure=TEST_NETWORK_INFRASTRUCTURE_FILENAME,
                                         input_pipeline=self.input_pipeline)

        self.pipeline = Pipeline(self.input_pipeline).load()

        self.infrastructure = Infrastructure(TEST_INFRASTRUCTURE_FILENAME).load()
        self.net_infrastructure = NetworkInfrastructure(TEST_NETWORK_INFRASTRUCTURE_FILENAME).load()

        self.number_of_objectives = len(OBJECTIVES_LABELS)
        self.number_of_constraints = len(CONSTRAINT_LABELS)

        self.unconstrained_models = []
        with open(UNCONSTRAINED_INFRA_FILENAME, 'r') as f:
            for line in f.readlines():
                self.unconstrained_models.append(line.split(','))
        self.unconstrained_models = np.asfarray(self.unconstrained_models, dtype=bool)
        self.unconstrained_model_solution = BinarySolution(number_of_objectives=self.number_of_objectives,
                                                           number_of_constraints=self.number_of_constraints,
                                                           number_of_variables=self.infrastructure.shape[1])
        self.unconstrained_model_solution.variables = self.unconstrained_models.transpose()

        self.unconstrained_network = []
        with open(UNCONSTRAINED_NET_INFRA_FILENAME, 'r') as f:
            for line in f.readlines():
                self.unconstrained_network.append(line.split(','))
        self.unconstrained_network = np.asfarray(self.unconstrained_network, dtype=bool)
        self.unconstrained_network_solution = BinarySolution(number_of_objectives=self.number_of_objectives,
                                                             number_of_constraints=self.number_of_constraints,
                                                             number_of_variables=self.net_infrastructure.shape[1])
        self.unconstrained_network_solution.variables = self.unconstrained_network.transpose()

        self.unconstrained_solution = CompositeSolution([self.unconstrained_model_solution,
                                                         self.unconstrained_network_solution])

    def test_evaluate(self):
        solution = self.problem.evaluate(self.unconstrained_solution)

        expected_objectives = [-1*0.485, np.round(0.299996, 3), 0.5, 0.5]
        self.assertListEqual(expected_objectives, solution.objectives)

        expected_constraints = [0, 0, 0, 0, 0, 0, 0, 0]
        self.assertListEqual(expected_constraints, solution.constraints)

    def test_create_solution(self):
        solution = self.problem.create_solution()

        self.assertEqual(5, len(solution.variables[0].variables))
        self.assertEqual(10, len(solution.variables[0].variables[0]))

        self.assertEqual(5, len(solution.variables[1].variables))
        self.assertEqual(10, len(solution.variables[1].variables[0]))

        self.assertEqual(CompositeSolution, type(solution))

    def test_get_name(self):
        self.assertEqual("DeploymentProblem", self.problem.get_name())

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(self('test_evaluate'))
        suite.addTest(self('test_create_solution'))
        suite.addTest(self('test_get_name'))
        return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(TestProblem().suite())
