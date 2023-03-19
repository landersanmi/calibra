#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import spatial

from src.core.models.infrastructure import Infrastructure
from src.core.models.network_infrastructure import NetworkInfrastructure

from src.core.optimizer import Optimizer
from src.core.utils import (
    WriteObjectivesToFileObserver,
    WriteObjectivesToTensorboardObserver,
    ParetoTools,
    Evaluate,
)
from src.core.termination_criterions import (
    StoppingByNonDominance,
    StoppingByTotalDominance,
    StoppingByConstraintsMet,
    StoppingByFullPareto,
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
from jmetal.util.comparator import StrengthAndKNNDistanceComparator, RankingAndCrowdingDistanceComparator

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
        #termination_criterion=StoppingByTime(max_seconds=120),
        #termination_criterion=StoppingByConstraintsMet(tensorboard_logger),
        #termination_criterion=StoppingByGenerationsAfterConstraintsMet(generations=5, logger=tensorboard_logger),
        #termination_criterion=StoppingByTimeAfterConstraintsMet(max_seconds=10, logger=tensorboard_logger),
        termination_criterion=StoppingByTimeOrGenerationsAfterConstraintsMet(max_seconds=800, max_generations=20, logger=tensorboard_logger),
        observer=WriteObjectivesToTensorboardObserver(tensorboard_logger),
        population_size=population_size,
        #dominance_comparator=StrengthAndKNNDistanceComparator(),
        #dominance_comparator=RankingAndCrowdingDistanceComparator(),
    )
    o.run()

    front = o.get_front()
    for i, solution in enumerate(front):
        front[i].objectives[0] = abs(solution.objectives[0])

    plot_front = InteractivePlot(title="Pareto front approximation", axis_labels=OBJECTIVES_LABELS)
    plot_front.plot(front, label="", filename="tmp/plots/paretos/front_plot", normalize=False)

    objectives_and_constraints, objectives, device_solutions, net_solutions = [], [], [], []
    for s in o.get_front():
        objectives_and_constraints.append(s.objectives + s.constraints)
        objectives.append(s.objectives)
        device_solutions.append(s.variables[0].variables)
        net_solutions.append(s.variables[1].variables)

    print("Solution Attributes:")
    for solution in o.get_front():
        print(solution.objectives)

    df = pd.DataFrame(
        objectives_and_constraints,
        columns=SOLUTION_DF_COLUMNS,
    )

    objectives_df = pd.DataFrame(objectives, columns=OBJECTIVES_LABELS)
    row_dict = dict()
    for objective, utopian_value in zip(OBJECTIVES_LABELS, UTOPIAN_CASE):
        row_dict[objective] = utopian_value
    row = pd.Series(row_dict)
    objectives_df = pd.concat([objectives_df, row.to_frame().T], ignore_index= True)

    print(objectives_df.iloc[-1].values)
    objectives_df['cosine_similarity'] = objectives_df.apply(lambda row: spatial.distance.cosine(row.values, objectives_df.iloc[-1]), axis=1)
    objectives_df = objectives_df.sort_values(by=['cosine_similarity'], ascending=True)

    print(objectives_df.head())
    best_solutions = [objectives_df.head(2)[1:2].index]
    for i, col in enumerate(df.columns):
        if i < o.problem.number_of_objectives:
            print("--------Goals."+col+"---------")
            print(df.sort_values(by=[col]).head(1))
            best_solutions = np.append(best_solutions, df.sort_values(by=[col]).head(1).index)


    '''
    for index in best_solutions:
        print(index)
        device_names = Infrastructure(file_infrastructure).load().hostname.to_numpy()
        device_solution_df = pd.DataFrame(device_solutions[int(index)], columns=device_names)
        net_device_names = NetworkInfrastructure(file_net_infrastructure).load().id.to_numpy()
        net_solution_df = pd.DataFrame(net_solutions[int(index)], columns=device_names)

        device_matrix = np.array(device_solution_df[device_names].values, dtype=float).T
        net_matrix = np.array(net_solution_df[device_names].values, dtype=float).T
        models_indexes = device_solution_df.index

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,9), dpi=90, gridspec_kw={'width_ratios': [len(models_indexes), len(net_device_names)]})

        ax[0].imshow(device_matrix, cmap='GnBu', interpolation='nearest')
        plt.sca(ax[0])
        plt.yticks(range(device_matrix.shape[0]), device_names)
        plt.xticks(range(device_matrix.shape[1]), models_indexes)
        plt.xticks(rotation=30)
        plt.xlabel('Infra devices')
        plt.xlabel('Pipeline Models')
        plt.title("Device Solution")

        # place '0' or '1' values centered in the individual squares
        for x in range(device_matrix.shape[0]):
            for y in range(device_matrix.shape[1]):
                ax[0].annotate(str(device_matrix[x, y])[0], xy=(y, x),
                            horizontalalignment='center', verticalalignment='center')

        ax[1].imshow(net_matrix, cmap='GnBu', interpolation='nearest')
        plt.sca(ax[1])
        plt.yticks(range(net_matrix.shape[0]), device_names)
        plt.xticks(range(net_matrix.shape[1]), net_device_names, fontsize=7)
        plt.xticks(rotation=30)
        plt.xlabel('Infra devices')
        plt.xlabel('Network Devices')
        plt.title("Network Solution")

        # place '0' or '1' values centered in the individual squares
        for x in range(net_matrix.shape[0]):
            for y in range(net_matrix.shape[1]):
                ax[1].annotate(str(net_matrix[x, y])[0], xy=(y, x),
                         horizontalalignment='center', verticalalignment='center')
        plt.show()
    '''
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
    total_times = []
    pipelines = ['5NET', '5NETXL', '10NET', '10NETXL']
    for p in pipelines:
        file_pipeline = PIPELINE_FILENAME.format(pipeline=p)
        with open(file_pipeline, "r") as input_data_file:
            input_pipeline = input_data_file.read()
        
        population_size = 200
        pipe_time = []
        pipe_time.append(p)
        
        for i in range(5):
            tensorboard_logger = TensorboardLogger(algo_name=str(p) + '[' + str(i) + ']')
            start_time = time.time()
            LOGGER.info(f"Executing iteration {i} of {file_pipeline}.")
            Optimizer(
                file_infrastructure=file_infrastructure,
                file_net_infrastructure=file_net_infrastructure,
                input_pipeline=input_pipeline,
                termination_criterion=StoppingByConstraintsMet(tensorboard_logger),
                observer=WriteObjectivesToTensorboardObserver(tensorboard_logger),
                population_size=population_size,
            ).run()
            end_time = time.time()
            pipe_time.append(end_time - start_time)
        total_times.append(pipe_time)

    print(total_times)
    if os.path.exists(TIMES_FILENAME):
        os.remove(TIMES_FILENAME)
    with open(TIMES_FILENAME, "w", newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(total_times)


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
