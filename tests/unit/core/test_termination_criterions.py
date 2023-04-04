import unittest
from datetime import datetime
import numpy as np

from tests.unit.core.constants import (
    OBJECTIVES_LABELS,
    CONSTRAINT_LABELS,
    TEST_SMALL_NETWORK_INFRASTRUCTURE_FILENAME,
    TEST_SMALL_PIPELINE_FILENAME,
    TEST_SMALL_INFRASTRUCTURE_FILENAME,
    UNCONSTRAINED_INFRA_FILENAME,
    UNCONSTRAINED_NET_INFRA_FILENAME
)

from src.core.models.infrastructure import Infrastructure
from src.core.models.pipeline import Pipeline
from src.core.models.network_infrastructure import NetworkInfrastructure
from src.core.constraints import Constraints
from src.core.optimizer import Optimizer
from src.core.termination_criterions import (
    StoppingByConstraintsMet,
    StoppingByGenerationsAfterConstraintsMet,
    StoppingByTimeAfterConstraintsMet,
    StoppingByTimeOrGenerationsAfterConstraintsMet
)


class TestTerminationCriterions(unittest.TestCase):

    def setUp(self):
        self.file_pipeline = TEST_SMALL_PIPELINE_FILENAME.format(pipeline="5NET")
        with open(self.file_pipeline, "r") as input_data_file:
            self.input_pipeline = input_data_file.read()
        self.pipeline = Pipeline(self.input_pipeline).load()
        self.infrastructure = Infrastructure(TEST_SMALL_INFRASTRUCTURE_FILENAME).load()
        self.net_infrastructure = NetworkInfrastructure(TEST_SMALL_NETWORK_INFRASTRUCTURE_FILENAME).load()

    def test_stopping_by_constraints_met(self):
        self.optimizer = Optimizer(
            termination_criterion=StoppingByConstraintsMet(logger=None),
            file_infrastructure=TEST_SMALL_INFRASTRUCTURE_FILENAME,
            file_net_infrastructure=TEST_SMALL_NETWORK_INFRASTRUCTURE_FILENAME,
            input_pipeline=self.input_pipeline,
            population_size=10
        )

        self.optimizer.run()

        self.assertEqual(self.optimizer.termination_criterion.is_met, True)
        for solution in self.optimizer.get_front():
            c = Constraints(solution, self.infrastructure, self.net_infrastructure, self.pipeline)
            self.assertEqual(c.cpu_constraint(), 0)
            self.assertEqual(c.deployment_constraint(), 0)
            self.assertEqual(c.ram_constraint(), 0)
            self.assertEqual(c.bandwidth_constraint(), 0)
            self.assertEqual(c.net_deployment_constraint(), 0)
            self.assertEqual(c.net_device_capacity_constraint(), 0)
            self.assertEqual(c.net_traffic_capacity_constraint(), 0)
            self.assertEqual(c.net_layers_constraint(), 0)

    def test_stopping_by_generations_after_constraints_met(self):
        max_generations = 15
        self.optimizer = Optimizer(
            termination_criterion=StoppingByGenerationsAfterConstraintsMet(generations=max_generations,
                                                                           logger=None),
            file_infrastructure=TEST_SMALL_INFRASTRUCTURE_FILENAME,
            file_net_infrastructure=TEST_SMALL_NETWORK_INFRASTRUCTURE_FILENAME,
            input_pipeline=self.input_pipeline,
            population_size=10
        )

        self.optimizer.run()

        self.assertEqual(self.optimizer.termination_criterion.generations_after_constraints, 15)
        self.assertEqual(self.optimizer.termination_criterion.is_met, True)
        for solution in self.optimizer.get_front():
            c = Constraints(solution, self.infrastructure, self.net_infrastructure, self.pipeline)
            self.assertEqual(c.cpu_constraint(), 0)
            self.assertEqual(c.deployment_constraint(), 0)
            self.assertEqual(c.ram_constraint(), 0)
            self.assertEqual(c.bandwidth_constraint(), 0)
            self.assertEqual(c.net_deployment_constraint(), 0)
            self.assertEqual(c.net_device_capacity_constraint(), 0)
            self.assertEqual(c.net_traffic_capacity_constraint(), 0)
            self.assertEqual(c.net_layers_constraint(), 0)

    def test_stopping_by_time_after_constraints_met(self):
        max_seconds = 2
        self.optimizer = Optimizer(
            termination_criterion=StoppingByTimeAfterConstraintsMet(max_seconds=max_seconds,
                                                                    logger=None),
            file_infrastructure=TEST_SMALL_INFRASTRUCTURE_FILENAME,
            file_net_infrastructure=TEST_SMALL_NETWORK_INFRASTRUCTURE_FILENAME,
            input_pipeline=self.input_pipeline,
            population_size=10
        )

        start_time = datetime.now()
        self.optimizer.run()
        total_time = datetime.now() - start_time
        total_time = total_time.seconds - self.optimizer.termination_criterion.seconds_to_met_constraints

        self.assertEqual(np.ceil(total_time), 2)
        for solution in self.optimizer.get_front():
            c = Constraints(solution, self.infrastructure, self.net_infrastructure, self.pipeline)
            self.assertEqual(c.cpu_constraint(), 0)
            self.assertEqual(c.deployment_constraint(), 0)
            self.assertEqual(c.ram_constraint(), 0)
            self.assertEqual(c.bandwidth_constraint(), 0)
            self.assertEqual(c.net_deployment_constraint(), 0)
            self.assertEqual(c.net_device_capacity_constraint(), 0)
            self.assertEqual(c.net_traffic_capacity_constraint(), 0)
            self.assertEqual(c.net_layers_constraint(), 0)

    def test_stopping_by_time_or_generations_after_constraints_met(self):
        max_seconds = 2
        max_generations = 1000
        self.optimizer = Optimizer(
            termination_criterion=StoppingByTimeOrGenerationsAfterConstraintsMet(max_seconds=max_seconds,
                                                                                 max_generations=max_generations,
                                                                                 logger=None),
            file_infrastructure=TEST_SMALL_INFRASTRUCTURE_FILENAME,
            file_net_infrastructure=TEST_SMALL_NETWORK_INFRASTRUCTURE_FILENAME,
            input_pipeline=self.input_pipeline,
            population_size=10
        )

        start_time = datetime.now()
        self.optimizer.run()
        total_time = datetime.now() - start_time
        total_time = total_time.seconds - self.optimizer.termination_criterion.seconds_to_met_constraints
        generations_after_constraints_met = self.optimizer.termination_criterion.generations_after_constraints
        if np.ceil(total_time) == max_seconds or generations_after_constraints_met == max_generations:
            conditions_met = True
        else:
            conditions_met = False

        self.assertEqual(conditions_met, True)
        self.assertEqual(self.optimizer.termination_criterion.is_met, True)
        for solution in self.optimizer.get_front():
            c = Constraints(solution, self.infrastructure, self.net_infrastructure, self.pipeline)
            self.assertEqual(c.cpu_constraint(), 0)
            self.assertEqual(c.deployment_constraint(), 0)
            self.assertEqual(c.ram_constraint(), 0)
            self.assertEqual(c.bandwidth_constraint(), 0)
            self.assertEqual(c.net_deployment_constraint(), 0)
            self.assertEqual(c.net_device_capacity_constraint(), 0)
            self.assertEqual(c.net_traffic_capacity_constraint(), 0)
            self.assertEqual(c.net_layers_constraint(), 0)
            
    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(self('test_stopping_by_constraints_met'))
        suite.addTest(self('test_stopping_by_generations_after_constraints_met'))
        suite.addTest(self('test_stopping_by_time_after_constraints_met'))
        suite.addTest(self('test_stopping_by_time_or_generations_after_constraints_met'))
        return suite
        
        
if __name__ == "__main__":
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(TestProblem().suite())




