import numpy.ma.core

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

from src.core.objectives import Objectives
from src.core.models.infrastructure import Infrastructure
from src.core.models.pipeline import Pipeline
from src.core.models.network_infrastructure import NetworkInfrastructure


class TestObjectives(unittest.TestCase):

    def setUp(self):
        self.file_pipeline = TEST_PIPELINE_FILENAME.format(pipeline="5NET")
        with open(self.file_pipeline, "r") as input_data_file:
            self.input_pipeline = input_data_file.read()
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

    def test_get_consumption(self):
        consumption = Objectives().get_consumption(self.pipeline, self.infrastructure, self.unconstrained_solution)
        self.assertEqual(consumption, np.round(0.299996, 3))  # (0.19999 + 0.39999 + 0.59999 + 0.79999 + 1) / 10 -> Normalized values

        modified_solution = self.unconstrained_solution
        modified_solution.variables[0].variables[1][0] = 1
        consumption = Objectives().get_consumption(self.pipeline, self.infrastructure, modified_solution)
        self.assertEqual(consumption, np.round(0.299996, 3)) # Doesn´t affect due to max threads reached on device 0

        modified_solution.variables[0].variables[1][0] = 0
        modified_solution.variables[0].variables[0][0] = 0
        modified_solution.variables[0].variables[1][1] = 0
        modified_solution.variables[0].variables[2][2] = 0
        modified_solution.variables[0].variables[3][3] = 0
        modified_solution.variables[0].variables[4][4] = 0
        consumption = Objectives().get_consumption(self.pipeline, self.infrastructure, modified_solution)
        self.assertEqual(consumption, 0)

        modified_solution.variables[0].variables[4][1] = 1
        modified_solution.variables[0].variables[4][2] = 1
        consumption = Objectives().get_consumption(self.pipeline, self.infrastructure, modified_solution)
        self.assertEqual(consumption, 0.1)  # 1/10

    def test_get_performance(self):
        performance = Objectives().get_performance(self.pipeline, self.infrastructure, self.unconstrained_solution)
        self.assertEqual(performance, 0.485)  # (0.25*0.99 + 0.5*0.98 + 0.75*0.97 + 1*0.96) / 5 -> Normalized

        modified_solution = self.unconstrained_solution
        modified_solution.variables[0].variables[1][0] = 1
        performance = Objectives().get_performance(self.pipeline, self.infrastructure, modified_solution)
        self.assertEqual(performance, 0.46025)  # (0.25*0.99/2 + 0.5*0.98 + 0.75*0.97 + 1*0.96) / 5 -> Normalized

        modified_solution.variables[0].variables[1][0] = 0
        modified_solution.variables[0].variables[0][0] = 0
        modified_solution.variables[0].variables[1][1] = 0
        modified_solution.variables[0].variables[2][2] = 0
        modified_solution.variables[0].variables[3][3] = 0
        modified_solution.variables[0].variables[4][4] = 0
        performance = Objectives().get_performance(self.pipeline, self.infrastructure, modified_solution)
        not_numeric = False
        try:
            int(performance)
        except:
            not_numeric = True
        self.assertEqual(not_numeric, True)

        modified_solution.variables[0].variables[1][4] = 1
        modified_solution.variables[0].variables[2][4] = 1
        performance = Objectives().get_performance(self.pipeline, self.infrastructure, modified_solution)
        self.assertEqual(performance, 0.384)  # (1*0.96*2) / 5 -> Normalized

    def test_get_net_cost(self):
        net_cost = Objectives().get_net_cost(self.net_infrastructure, self.unconstrained_solution)
        self.assertEqual(net_cost, 0.5) # 0.25 + 0.5 + 0.75 + 1 / 5 -> Normalized

        modified_solution = self.unconstrained_solution
        modified_solution.variables[1].variables[1][0] = 1
        net_cost = Objectives().get_net_cost(self.net_infrastructure, modified_solution)
        self.assertEqual(net_cost, 0.5)  # Same cost because the net device was already in use

        modified_solution = self.unconstrained_solution
        modified_solution.variables[1].variables[1][0] = 0
        modified_solution.variables[1].variables[4][4] = 0
        net_cost = Objectives().get_net_cost(self.net_infrastructure, modified_solution)
        self.assertEqual(net_cost, 0.3)  # 0.25 + 0.5 + 0.75 / 5 -> Normalized

    def test_get_net_fail_probability(self):
        net_fail_probability = Objectives().get_net_fail_probability(self.net_infrastructure, self.unconstrained_solution)
        self.assertEqual(net_fail_probability, 0.5) # 0.01 + 0.5 + 0.75 + 1 / 5 -> Normalized

        modified_solution = self.unconstrained_solution
        modified_solution.variables[1].variables[4][0] = 1
        net_fail_probability = Objectives().get_net_fail_probability(self.net_infrastructure, modified_solution)
        self.assertEqual(net_fail_probability, 0.5)  # Same failure prob. because the net device was already in use

        modified_solution = self.unconstrained_solution
        modified_solution.variables[1].variables[4][0] = 0
        modified_solution.variables[1].variables[0][0] = 0
        modified_solution.variables[1].variables[2][2] = 0
        modified_solution.variables[1].variables[4][4] = 0
        net_fail_probability = Objectives().get_net_fail_probability(self.net_infrastructure, modified_solution)
        self.assertEqual(net_fail_probability, 0.2)  # 0.25 + 0.75 / 5 -> Normalized

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(self('test_get_consumption'))
        suite.addTest(self('test_get_performance'))
        suite.addTest(self('test_get_net_cost'))
        suite.addTest(self('test_get_net_fail_probability'))
        return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(TestObjectives().suite())
