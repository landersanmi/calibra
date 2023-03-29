from tests.unit.core.constants import (
    TEST_NETWORK_INFRASTRUCTURE_FILENAME,
    TEST_PIPELINE_FILENAME,
    TEST_INFRASTRUCTURE_FILENAME,
    UNCONSTRAINED_INFRA_FILENAME,
    UNCONSTRAINED_NET_INFRA_FILENAME
)

import unittest
import numpy as np
from copy import copy
from src.core.constraints import Constraints
from src.core.models.infrastructure import Infrastructure
from src.core.models.pipeline import Pipeline
from src.core.models.network_infrastructure import NetworkInfrastructure

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning)

from jmetal.core.solution import CompositeSolution, BinarySolution


class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.file_pipeline = TEST_PIPELINE_FILENAME.format(pipeline="5NET")
        with open(self.file_pipeline, "r") as input_data_file:
            self.input_pipeline = input_data_file.read()
        self.pipeline = Pipeline(self.input_pipeline).load()

        self.infrastructure = Infrastructure(TEST_INFRASTRUCTURE_FILENAME).load()
        self.net_infrastructure = NetworkInfrastructure(TEST_NETWORK_INFRASTRUCTURE_FILENAME).load()

        self.unconstrained_models = []
        with open(UNCONSTRAINED_INFRA_FILENAME, 'r') as f:
            for line in f.readlines():
                self.unconstrained_models.append(line.split(','))
        self.unconstrained_models = np.asfarray(self.unconstrained_models, dtype=bool)
        self.unconstrained_model_solution = BinarySolution(number_of_objectives=4,
                                                           number_of_constraints=8,
                                                           number_of_variables=self.infrastructure.shape[1])
        self.unconstrained_model_solution.variables = self.unconstrained_models.transpose()

        self.unconstrained_network = []
        with open(UNCONSTRAINED_NET_INFRA_FILENAME, 'r') as f:
            for line in f.readlines():
                self.unconstrained_network.append(line.split(','))
        self.unconstrained_network = np.asfarray(self.unconstrained_network, dtype=bool)
        self.unconstrained_network_solution = BinarySolution(number_of_objectives=4,
                                                           number_of_constraints=8,
                                                           number_of_variables=self.net_infrastructure.shape[1])
        self.unconstrained_network_solution.variables = self.unconstrained_network.transpose()

        self.unconstrained_solution = CompositeSolution([self.unconstrained_model_solution,
                                                         self.unconstrained_network_solution])

    def test_cpu_constraint_unmet(self):
        constrained_pipeline = self.pipeline.copy()
        constrained_pipeline.at[0, 'cpus'] = 100
        c = Constraints(self.unconstrained_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.cpu_constraint(), -100)  # -100/1

        constrained_pipeline.at[3, 'cpus'] = 200
        c = Constraints(self.unconstrained_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.cpu_constraint(), -150)  # -100/1 -200/4

        constrained_cpu_solution = self.unconstrained_solution
        constrained_cpu_solution.variables[0].variables[4] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        c = Constraints(constrained_cpu_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(round(c.cpu_constraint(), 5), round(-162.4166667, 5))  # -105/1, -7/2, -8/3(2.66667), -205/4(51.25)

    def test_deployment_constraint_unmet(self):
        constrained_deployment_solution = self.unconstrained_solution
        constrained_deployment_solution.variables[0].variables[0][0] = 0
        c = Constraints(constrained_deployment_solution, self.infrastructure, self.net_infrastructure, self.pipeline)
        self.assertEqual(c.deployment_constraint(), -1)

        constrained_deployment_solution.variables[0].variables[1][1] = 0
        constrained_deployment_solution.variables[0].variables[2][2] = 0
        c = Constraints(constrained_deployment_solution, self.infrastructure, self.net_infrastructure, self.pipeline)
        self.assertEqual(c.deployment_constraint(), -3)

    def test_ram_constraint_unmet(self):
        constrained_pipeline = self.pipeline.copy()
        constrained_pipeline.at[0, 'memory'] = 10
        c = Constraints(self.unconstrained_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.ram_constraint(), -10)  # -10/1

        constrained_pipeline.at[3, 'memory'] = 20
        c = Constraints(self.unconstrained_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.ram_constraint(), -15)  # -10/1 -20/4

        constrained_ram_solution = self.unconstrained_solution
        constrained_ram_solution.variables[0].variables[0] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        c = Constraints(constrained_ram_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(round(c.ram_constraint(), 5), round(-30.83333334, 5))  # -10/1, -10/2, -10/3(3.33334), -30/4(7.5), -10/5

    def test_bandwidth_constraint_unmet(self):
        constrained_pipeline = self.pipeline.copy()
        constrained_pipeline.at[1, 'network'] = 1000
        c = Constraints(self.unconstrained_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.bandwidth_constraint(), -980)  # 20-1000

        constrained_pipeline.at[4, 'network'] = 5000
        c = Constraints(self.unconstrained_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.bandwidth_constraint(), -5930)  # 20-1000, 30-40, 40-50, 50-5000

        constrained_bandwidth_solution = self.unconstrained_solution
        constrained_bandwidth_solution.variables[0].variables[0] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        c = Constraints(constrained_bandwidth_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.bandwidth_constraint(), -5970)  # 20-1010, 30-40, 40-50, 50-5010


    def test_net_deployment_constraint_unmet(self):
        constrained_net_deployment_solution = self.unconstrained_solution
        constrained_net_deployment_solution.variables[1].variables[0][4] = 1
        c = Constraints(constrained_net_deployment_solution, self.infrastructure, self.net_infrastructure, self.pipeline)
        self.assertEqual(c.net_deployment_constraint(), -1)

        constrained_net_deployment_solution.variables[1].variables[0][4] = 0
        constrained_net_deployment_solution.variables[0].variables[4][4] = 0
        c = Constraints(constrained_net_deployment_solution, self.infrastructure, self.net_infrastructure, self.pipeline)
        self.assertEqual(c.net_deployment_constraint(), -1)

        constrained_net_deployment_solution.variables[0].variables[4][4] = 1
        constrained_net_deployment_solution.variables[1].variables[0][0] = 0
        c = Constraints(constrained_net_deployment_solution, self.infrastructure, self.net_infrastructure, self.pipeline)
        self.assertEqual(c.net_deployment_constraint(), -1)

        constrained_net_deployment_solution.variables[1].variables[1][1] = 0
        constrained_net_deployment_solution.variables[1].variables[2][2] = 0
        c = Constraints(constrained_net_deployment_solution, self.infrastructure, self.net_infrastructure, self.pipeline)
        self.assertEqual(c.net_deployment_constraint(), -3)


    def test_net_device_capacity_constraint_unmet(self):
        constrained_net_dev_capacity_solution = self.unconstrained_solution
        constrained_net_dev_capacity_solution.variables[1].variables[0][1] = 1
        constrained_net_dev_capacity_solution.variables[1].variables[0][2] = 1
        c = Constraints(constrained_net_dev_capacity_solution, self.infrastructure, self.net_infrastructure, self.pipeline)
        self.assertEqual(c.net_device_capacity_constraint(), -2)

        constrained_net_dev_capacity_solution.variables[1].variables[0][1] = 1
        constrained_net_dev_capacity_solution.variables[1].variables[0][2] = 1
        constrained_net_dev_capacity_solution.variables[1].variables[4][0] = 1
        constrained_net_dev_capacity_solution.variables[1].variables[4][1] = 1
        constrained_net_dev_capacity_solution.variables[1].variables[4][2] = 1
        constrained_net_dev_capacity_solution.variables[1].variables[4][3] = 1
        constrained_net_dev_capacity_solution.variables[1].variables[4][5] = 1
        c = Constraints(constrained_net_dev_capacity_solution, self.infrastructure, self.net_infrastructure, self.pipeline)
        self.assertEqual(c.net_device_capacity_constraint(), -3)

    def test_net_traffic_capacity_constraint_unmet(self):
        constrained_pipeline = self.pipeline.copy()
        constrained_pipeline.at[0, 'network'] = 1000
        c = Constraints(self.unconstrained_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.net_traffic_capacity_constraint(), -990)

        constrained_bandwidth_solution = self.unconstrained_solution
        constrained_bandwidth_solution.variables[1].variables[0] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        c = Constraints(constrained_bandwidth_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.net_traffic_capacity_constraint(), -1130)  # 10-1140

    def test_net_layers_constraint_unmet(self):
        constrained_pipeline = self.pipeline.copy()
        constrained_pipeline.at[3, 'layer'] = "cloud"
        constrained_pipeline.at[4, 'layer'] = "premises"
        c = Constraints(self.unconstrained_solution, self.infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.net_layers_constraint(), -2)

        constrained_infrastructure = self.infrastructure.copy()
        constrained_infrastructure.at[3, 'cloud_type'] = "edge"
        constrained_infrastructure.at[4, 'cloud_type'] = "edge"
        c = Constraints(self.unconstrained_solution, constrained_infrastructure, self.net_infrastructure, constrained_pipeline)
        self.assertEqual(c.net_layers_constraint(), -4)

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(self('test_cpu_constraint_unmet'))
        suite.addTest(self('test_deployment_constraint_unmet'))
        suite.addTest(self('test_ram_constraint_unmet'))
        suite.addTest(self('test_bandwidth_constraint_unmet'))
        suite.addTest(self('test_net_deployment_constraint_unmet'))
        suite.addTest(self('test_net_device_capacity_constraint_unmet'))
        suite.addTest(self('test_net_traffic_capacity_constraint_unmet'))
        suite.addTest(self('test_net_layers_constraint_unmet'))
        return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(TestConstraints().suite())
