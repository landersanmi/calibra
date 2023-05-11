# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.deployment_report import DeploymentReport  # noqa: E501
from swagger_server.test import BaseTestCase


class TestOptimizationController(BaseTestCase):
    """OptimizationController integration test stubs"""

    def test_post_optimization(self):
        """Test case for post_optimization

        Post optimization
        """
        data = dict(pipeline='pipeline_example',
                    computing_infra='computing_infra_example',
                    network_infra='network_infra_example',
                    population_size=56,
                    generations_check=True,
                    max_generations=56,
                    time_check=True,
                    max_time=56)
        response = self.client.open(
            '/api/v1/optimize',
            method='POST',
            data=data,
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
