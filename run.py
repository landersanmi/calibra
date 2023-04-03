#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import logging
import os
import numpy as np

from src.core.optimizer import Optimizer
from src.core.utils import (
    WriteObjectivesToFileObserver,
    WriteObjectivesToTensorboardObserver,
    ParetoTools,
    Evaluate,
)
from src.core.termination_criterions import (
    StoppingByConstraintsMet,
    StoppingByGenerationsAfterConstraintsMet,
    StoppingByTimeAfterConstraintsMet,
    StoppingByTimeOrGenerationsAfterConstraintsMet,
)
from src.core.constants import (
    PIPELINE_FILENAME,
    SOLUTION_DF_COLUMNS,
    OBJECTIVES_LABELS,
    UTOPIAN_CASE,
    TIMES_FILENAME,
    INFRASTRUCTURE_FILENAME,
    NETWORK_INFRASTRUCTURE_FILENAME,
)
from src.core.tensorboard_logger import TensorboardLogger

from jmetal.util.termination_criterion import StoppingByEvaluations, StoppingByTime
from jmetal.lab.visualization import InteractivePlot

LOGGER = logging.getLogger("optimizer")


def compete(file_infrastructure: str, file_net_infrastructure:str, pipeline: str):
    file_pipeline = PIPELINE_FILENAME.format(pipeline=pipeline)
    population_size = 200

    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    tensorboard_logger = TensorboardLogger(algo_name=str(pipeline))
    o = Optimizer(
        file_infrastructure=file_infrastructure,
        file_net_infrastructure=file_net_infrastructure,
        input_pipeline=input_pipeline,
        termination_criterion=StoppingByTimeOrGenerationsAfterConstraintsMet(max_seconds=5, max_generations=200, logger=tensorboard_logger),
        observer=WriteObjectivesToTensorboardObserver(tensorboard_logger),
        population_size=population_size,
    )
    o.run()

    front = o.get_front()
    for i, solution in enumerate(front):
        front[i].objectives[0] = abs(solution.objectives[0])
        print(front[i].objectives)

    plot_front = InteractivePlot(title="Pareto front approximation", axis_labels=OBJECTIVES_LABELS)
    plot_front.plot(front, label="", filename="tmp/plots/paretos/front_plot", normalize=False)
    plot_front.plot(front, label="", filename="tmp/plots/paretos/front_plot_normalized", normalize=True)

    print("Pareto Fronts values:")
    for solution in o.get_front():
        print(solution.objectives)

    best_solution = o.get_best_solution(include_fitness_specific=True)
    o.plot_deployments(best_solution)

    pt = ParetoTools(o.get_front())
    pt.save()


def evaluate_solution(file_solution: str):
    e = Evaluate(file_solution=file_solution)
    print(f"Constraints.CPU = {e.constraint_cpu()}")
    print(f"Constraints.RAM = {e.constraint_ram()}")
    print(f"Constraints.Deployment = {e.constraint_deployment()}")
    print(f"Constraints.Net Deployment = {e.constraint_net_deployment()}")
    print(f"Constraints.Net Device Capacity = {e.constraint_net_device_capacity()}")
    print(f"Constraints.Net Traffic Capacity = {e.constraint_net_traffic_capacity()}")
    print(f"Constraints.Net Layers = {e.constraint_net_layers()}")

    print(f"Goals.Model Performance = {e.model_performance()}")
    print(f"Goals.Cost = {e.cost()}")
    print(f"Goals.Network Cost = {e.network_cost()}")
    print(f"Goals.Network Fail Probability = {e.network_fail_probability()}")


def generate_times(file_infrastructure: str, file_net_infrastructure:str):
    pipelines = ['5NET', '10NET', '15NET', '20NET', '30NET']
    iterations = 7
    total_times = []

    if os.path.exists(TIMES_FILENAME):
        os.remove(TIMES_FILENAME)

    columnames = list()
    columnames.append("pipeline")
    for i in range(iterations):
        columnames.append(str(i))
    columnames.append("avg_time")
    columnames.append("std_deviation")
    for objective in OBJECTIVES_LABELS:
        columnames.append(objective.replace(" ", "_").lower())
    columnames.append("avg_opt_time")

    with open(TIMES_FILENAME, "w", newline='') as times_file:
        writer = csv.writer(times_file)
        writer.writerow(columnames)

    for p in pipelines:
        file_pipeline = PIPELINE_FILENAME.format(pipeline=p)
        with open(file_pipeline, "r") as input_data_file:
            input_pipeline = input_data_file.read()
        
        population_size = 200
        row = list()
        row.append(p)
        objectives = []
        opt_times = []

        for i in range(iterations):
            tensorboard_logger = TensorboardLogger(algo_name=str(p) + '[' + str(i) + ']')
            LOGGER.info(f"Executing iteration {i} of {file_pipeline}.")
            optimizer = Optimizer(
                file_infrastructure=file_infrastructure,
                file_net_infrastructure=file_net_infrastructure,
                input_pipeline=input_pipeline,
                termination_criterion=StoppingByTimeOrGenerationsAfterConstraintsMet(max_seconds=600, max_generations=100, logger=tensorboard_logger),
                observer=WriteObjectivesToTensorboardObserver(tensorboard_logger),
                population_size=population_size,
            )
            optimizer.run()

            row.append(optimizer.termination_criterion.seconds_to_met_constraints)
            best_sol = optimizer.get_best_solution(include_fitness_specific=False)[0]
            print("Best solution ID: ", best_sol)
            objectives.append(optimizer.get_front()[best_sol].objectives)
            opt_times.append(optimizer.termination_criterion.total_seconds)

        times = list([float(time) for time in row[1:]])
        row.append(str(sum(times)/int(iterations)))
        row.append(str(np.std(times)))

        objectives = np.transpose(objectives).sum(axis=1)/iterations
        for objective in objectives:
            row.append(str(objective))
        row.append(str(sum(opt_times)/iterations))

        with open(TIMES_FILENAME, "a", newline='') as times_file:
            writer = csv.writer(times_file)
            writer.writerow(row)

        total_times.append(row)


def generate_pareto(file_infrastructure):
    file_pipeline = PIPELINE_FILENAME.format(pipeline=40)
    population_size = 100
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    o = Optimizer(
        file_infrastructure=file_infrastructure,
        input_pipeline=input_pipeline,
        # termination_criterion=StoppingByFullPareto(offspring_size=population_size),
        termination_criterion=StoppingByTime(max_seconds=600),
        population_size=population_size,
    )
    o.run()
    pt = ParetoTools(o.get_front())
    pt.save()


def generate_fitnesses(file_infrastructure, file_net_infrastructure):
    file_pipeline = PIPELINE_FILENAME.format(pipeline='40NET')
    population_size = 50
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    Optimizer(
        file_infrastructure=file_infrastructure,
        file_net_infrastructure=file_net_infrastructure,
        input_pipeline=input_pipeline,
        # termination_criterion=StoppingByTotalDominance(idle_evaluations=100),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=population_size * 400
        ),
        observer=WriteObjectivesToFileObserver(),
        population_size=population_size,
    ).run()


def generate_memory(file_infrastructure, number_of_models):
    file_pipeline = PIPELINE_FILENAME.format(pipeline=number_of_models)
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    Optimizer(
        file_infrastructure=file_infrastructure,
        input_pipeline=input_pipeline,
        termination_criterion=StoppingByTime(max_seconds=30),
    ).run()


def main():
    text = "This application optimizes PADL defined analytic models in heterogeneous infrastructures."
    parser = argparse.ArgumentParser(description=text)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-p",
        "--pareto",
        action="store_true",
        help="Generate pareto front.",
        required=False,
    )
    required.add_argument(
        "-t",
        "--times",
        action="store_true",
        help="Generate time metrics",
        required=False,
    )
    required.add_argument(
        "-f",
        "--fitnesses",
        action="store_true",
        help="Generate fitnesses metrics",
        required=False,
    )
    required.add_argument(
        "-m",
        "--memory",
        type=str,
        default=None,
        help="Indicate number of models (e.g., 5, 10, 20, 40, 80)",
        required=False,
    )
    required.add_argument(
        "-e",
        "--evaluate",
        type=str,
        default=None,
        help="Indicate a csv solution",
        required=False,
    )
    required.add_argument(
        "-c",
        "--compete",
        type=str,
        default=None,
        help="Get best solutions for each goal",
        required=False,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )

    if args.times:
        generate_times(file_infrastructure=INFRASTRUCTURE_FILENAME,
                       file_net_infrastructure=NETWORK_INFRASTRUCTURE_FILENAME
        )

    if args.pareto:
        generate_pareto(
            file_infrastructure=INFRASTRUCTURE_FILENAME)

    if args.fitnesses:
        generate_fitnesses(
            file_infrastructure=INFRASTRUCTURE_FILENAME,
            file_net_infrastructure=NETWORK_INFRASTRUCTURE_FILENAME,
        )

    if args.memory:
        generate_memory(
            file_infrastructure=INFRASTRUCTURE_FILENAME,
            number_of_models=args.memory,
        )

    if args.evaluate:
        evaluate_solution(file_solution=args.evaluate)

    if args.compete:
        compete(
            file_infrastructure=INFRASTRUCTURE_FILENAME,
            file_net_infrastructure=NETWORK_INFRASTRUCTURE_FILENAME,
            pipeline=args.compete,
        )


if __name__ == "__main__":
    main()
