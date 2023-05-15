import datetime
import json

from flask import request

from swagger_server.models.deployment_report import DeploymentReport  # noqa: E501
from swagger_server.models.optimization import Optimization
from swagger_server import util
from swagger_server.ai.termination_criterions import (
    StoppingByConstraintsMet,
    StoppingByGenerationsAfterConstraintsMet,
    StoppingByTimeAfterConstraintsMet,
    StoppingByTimeOrGenerationsAfterConstraintsMet,
)
from swagger_server.ai.utils import WriteObjectivesToTensorboardObserver, CompositeSolutionEncoder
from swagger_server.ai.tensorboard_logger import TensorboardLogger
from swagger_server.ai.optimizer import Optimizer


def post_optimization():  # noqa: E501
    """Post optimization

    Upload three files and parameters to generate a deployment optimization report # noqa: E501

    :param id: 
    :type id: str
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
    id = request.form['id']
    pipeline = request.files['pipeline'].read()
    computing_infra = request.files['computing_infra'].read()
    network_infra = request.files['network_infra'].read()
    population_size = request.form['population_size']
    generations_check = request.form['generations_check']
    max_generations = request.form['max_generations']
    time_check = request.form['time_check']
    max_time = request.form['max_time']

    if generations_check == 'false':
        generations_check = False
    elif generations_check == 'true':
        generations_check = True
    if time_check == 'false':
        time_check = False
    elif time_check == 'true':
        time_check = True

    optimization = Optimization(id,
                                pipeline,
                                computing_infra,
                                network_infra,
                                int(population_size),
                                bool(generations_check),
                                int(max_generations),
                                bool(time_check),
                                int(max_time))

    tensorboard_logger = TensorboardLogger(algo_name=str(optimization.id), log_dir='tensorboard_logs')
    termination_criterion = StoppingByConstraintsMet(logger=tensorboard_logger)
    if optimization.generations_check and optimization.time_check:
        termination_criterion = StoppingByTimeOrGenerationsAfterConstraintsMet(max_seconds=optimization.max_time,
                                                                               max_generations=optimization.max_generations,
                                                                               logger=tensorboard_logger)
    elif optimization.generations_check:
        termination_criterion = StoppingByGenerationsAfterConstraintsMet(max_generations=optimization.max_generations,
                                                                         logger=tensorboard_logger)
    elif optimization.time_check:
        termination_criterion = StoppingByTimeAfterConstraintsMet(max_seconds=optimization.max_time,
                                                                  logger=tensorboard_logger)

    o = Optimizer(
        file_infrastructure=optimization.computing_infra,
        file_net_infrastructure=optimization.network_infra,
        input_pipeline=optimization.pipeline,
        termination_criterion= termination_criterion,
        observer=WriteObjectivesToTensorboardObserver(tensorboard_logger),
        population_size=optimization.population_size
    )
    o.run()
    best_solutions_ids = o.get_best_solution(include_fitness_specific=True)
    solutions = o.get_best_solution_values(best_solutions_ids)

    id = optimization.id
    pareto_front_size = len(o.get_front())
    total_time = int(o.termination_criterion.total_seconds)
    time_to_met_constraints = int(o.termination_criterion.seconds_to_met_constraints)
    num_models = 0
    num_computing_devices = 0
    num_net_devices = 0
    report_date = str(datetime.datetime.now())
    best_solution = json.dumps(solutions[0], cls=CompositeSolutionEncoder)
    best_sol_performance_fitness = json.dumps(solutions[1], cls=CompositeSolutionEncoder)
    best_sol_cost_fitness = json.dumps(solutions[2], cls=CompositeSolutionEncoder)
    best_sol_net_cost_fitness = json.dumps(solutions[3], cls=CompositeSolutionEncoder)
    best_sol_net_fail_prob_fitness = json.dumps(solutions[4], cls=CompositeSolutionEncoder)
    print(best_solution)
    population_size = optimization.population_size
    max_generations = optimization.max_generations
    max_time = optimization.max_time

    result = DeploymentReport(id,
                              pareto_front_size,
                              total_time,
                              time_to_met_constraints,
                              num_models,
                              num_computing_devices,
                              num_net_devices,
                              report_date,
                              best_solution,
                              best_sol_performance_fitness,
                              best_sol_cost_fitness,
                              best_sol_net_cost_fitness,
                              best_sol_net_fail_prob_fitness,
                              population_size,
                              max_generations,
                              max_time
                              ).to_dict()
    return result
