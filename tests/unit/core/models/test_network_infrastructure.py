import unittest

import numpy as np

from tests.unit.core.constants import (
    TEST_NETWORK_INFRASTRUCTURE_FILENAME
)

from src.core.models.network_infrastructure import NetworkInfrastructure

class TestNetworkInfrastructure(unittest.TestCase):
    def setUp(self):
        try:
            self.net_infrastructure = NetworkInfrastructure(TEST_NETWORK_INFRASTRUCTURE_FILENAME).load()
        except:
            self.net_infrastructure = None

    def test_load(self):
        self.assertIsNotNone(self.net_infrastructure)

    def test_net_infrastructure_shape(self):
        self.assertEqual(5, self.net_infrastructure.shape[0])
        self.assertEqual(8, self.net_infrastructure.shape[1])

    def test_values(self):
        self.assertEqual(3, self.net_infrastructure.iloc[2]['id'])
        self.assertEqual("test4", self.net_infrastructure.iloc[3]['device_name'])
        self.assertEqual("Router", self.net_infrastructure.iloc[0]['device_type'])
        self.assertEqual(2, self.net_infrastructure.iloc[1]['max_devices'])
        self.assertEqual(50, self.net_infrastructure.iloc[4]['max_network_traffic'])
        self.assertEqual(1, max(self.net_infrastructure['cost'].values))
        self.assertEqual(0, min(self.net_infrastructure['failure_prob'].values))
        self.assertEqual("cloud", self.net_infrastructure.iloc[4]['layer'])

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(self('test_load'))
        suite.addTest(self('test_net_infrastructure_shape'))
        suite.addTest(self('test_values'))
        return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(TestNetworkInfrastructure().suite())
