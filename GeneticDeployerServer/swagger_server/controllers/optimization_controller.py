import connexion
import six

from swagger_server.models.deployment_report import DeploymentReport  # noqa: E501
from swagger_server import util


def post_optimization(pipeline, computing_infra, network_infra, population_size, generations_check, max_generations, time_check, max_time):  # noqa: E501
    """Post optimization

    Upload three files and parameters to generate a deployment optimization report # noqa: E501

    :param pipeline: 
    :type pipeline: strstr
    :param computing_infra: 
    :type computing_infra: strstr
    :param network_infra: 
    :type network_infra: strstr
    :param population_size: 
    :type population_size: int
    :param generations_check: 
    :type generations_check: bool
    :param max_generations: 
    :type max_generations: int
    :param time_check: 
    :type time_check: bool
    :param max_time: 
    :type max_time: int

    :rtype: DeploymentReport
    """
    return 'do some magic!'
