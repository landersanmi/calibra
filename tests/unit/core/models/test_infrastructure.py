import unittest

import numpy as np

from tests.unit.core.constants import (
    TEST_INFRASTRUCTURE_FILENAME,
)

from src.core.models.infrastructure import Infrastructure


class TestInfrastructure(unittest.TestCase):
    def setUp(self):
        try:
            self.infrastructure = Infrastructure(TEST_INFRASTRUCTURE_FILENAME).load()
            print("Not exceoted")
        except:
            pass

    def test_load(self):
        self.assertIsNotNone(self.infrastructure)

    def test_infrastructure_shape(self):
        self.assertEqual(10, self.infrastructure.shape[0])
        self.assertEqual(14, self.infrastructure.shape[1])

    def test_values(self):
        self.assertEqual("test8", self.infrastructure.iloc[8]['hostname'])
        self.assertEqual(1, self.infrastructure.iloc[0]['thread_count'])
        self.assertEqual(1, self.infrastructure.iloc[0]['memory'])
        self.assertEqual(50, self.infrastructure.iloc[4]['resillience'])
        self.assertEqual([1, 0.99, 0.98], self.infrastructure.iloc[2]['parallelization'].tolist())
        self.assertEqual(1, max(self.infrastructure['performance'].values))
        self.assertEqual(0, min(self.infrastructure['performance'].values))

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(self('test_load'))
        suite.addTest(self('test_infrastructure_shape'))
        suite.addTest(self('test_values'))
        return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(TestInfrastructure().suite())
