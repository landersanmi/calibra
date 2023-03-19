import unittest

from tests.unit.core.constants import (
    TEST_PIPELINE_FILENAME,
)

from src.core.models.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.file_pipeline = TEST_PIPELINE_FILENAME.format(pipeline="5NET")
        try:
            with open(self.file_pipeline, "r") as input_data_file:
                self.input_pipeline = input_data_file.read()
            self.pipeline = Pipeline(self.input_pipeline).load()
        except:
            self.pipeline = None

    def test_load(self):
        self.assertIsNotNone(self.pipeline)

    def test_pipeline_shape(self):
        self.assertEqual(5, self.pipeline.shape[0])
        self.assertEqual(4, self.pipeline.shape[1])

    def test_values(self):
        self.assertEqual(1, self.pipeline.iloc[0].cpus)
        self.assertEqual(1, self.pipeline.iloc[0].memory)
        self.assertEqual(10, self.pipeline.iloc[0].network)
        self.assertEqual(20, self.pipeline.iloc[1].network)
        self.assertEqual(3, self.pipeline.iloc[0].link)

    def suite(self):
        suite = unittest.TestSuite()
        suite.addTest(self('test_load'))
        suite.addTest(self('test_pipeline_shape'))
        suite.addTest(self('test_values'))
        return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(TestPipeline().suite())
