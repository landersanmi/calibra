#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import pandas as pd
import time
from src.core.optimizer import Optimizer
from src.core.utils import (
    StoppingByNonDominance,
    StoppingByTotalDominance,
    StoppingByConstraintsMet,
    StoppingByFullPareto,
    WriteObjectivesToFileObserver,
    ParetoTools,
    Evaluate,
)
from jmetal.util.termination_criterion import StoppingByEvaluations, StoppingByTime
from tabulate import tabulate

LOGGER = logging.getLogger("optimizer")


def compete(file_infrastructure: str, file_net_infrastructure:str, file_latencies: str, pipeline: str):
    file_pipeline = f"src/resources/pipeline_{pipeline}.yml"
    # population_size = 180
    population_size = 200
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    o = Optimizer(
        file_infrastructure=file_infrastructure,
        file_net_infrastructure=file_net_infrastructure,
        file_latencies=file_latencies,
        input_pipeline=input_pipeline,
        #termination_criterion=StoppingByTime(max_seconds=600),
        termination_criterion=StoppingByConstraintsMet(),
        observer=WriteObjectivesToFileObserver(),
        population_size=population_size,
    )
    o.run()

    objectives = []
    for s in o.get_front():
        objectives.append(s.objectives + s.constraints)

    columnames = [
            #"Resilience",
            "Model Perf",
            "Cost",
            "Network Perf",
            "Net Cost",
            "Net Fail Prob",
            "cpu",
            "ram",
            "deploy",
            #"privacy",
            "net deployment",
            "net device capacity",
            "net traffic capacity",
            "net layers"
        ]
    df = pd.DataFrame(
        objectives,
        columns=columnames,
    )

    for i, col in enumerate(df.columns):
        if i < o.problem.number_of_objectives:
            print("--------Goals."+col+"---------")
            print(df.sort_values(by=[col]).head(1))

    pt = ParetoTools(o.get_front())
    pt.save()
    '''
    print(
        f"Goals.Resilience = {tabulate(df.sort_values(by=['Resilience']).head(1), headers='keys', tablefmt='psql')}"
    )
    print(
        f"Goals.Model Performance = {tabulate(df.sort_values(by=['Model Perf']).head(1), headers='keys', tablefmt='psql')}"
    )
    print(
        f"Goals.Cost = {tabulate(df.sort_values(by=['Cost']).head(1), headers='keys', tablefmt='psql')}"
    )
    print(
        f"Goals.Network Performance = {tabulate(df.sort_values(by=['Network Perf']).head(1), headers='keys', tablefmt='psql')}"
    )
    print(
        f"Goals.Network Cost = {tabulate(df.sort_values(by=['Net Cost']).head(1), headers='keys', tablefmt='psql')}"
    )
    print(
        f"Goals.Network Fail Prob = {tabulate(df.sort_values(by=['Net Fail Prob']).head(1), headers='keys', tablefmt='psql')}"
    )
    '''


def evaluate_solution(file_solution: str):
    e = Evaluate(file_solution=file_solution)
    print(f"Constraints.CPU = {e.constraint_cpu()}")
    print(f"Constraints.RAM = {e.constraint_ram()}")
    print(f"Constraints.GPU = 0")
    print(f"Constraints.Deployment = {e.constraint_deployment()}")
    print(f"Constraints.Privacy = {e.constraint_privacy()}")
    print(f"Constraints.Net Deployment = {e.constraint_net_deployment()}")
    print(f"Constraints.Net Device Capacity = {e.constraint_net_device_capacity()}")
    print(f"Constraints.Net Traffic Capacity = {e.constraint_net_traffic_capacity()}")
    print(f"Constraints.Net Layers = {e.constraint_net_layers()}")

    print(f"Goals.Resilience = {e.resilience()}")
    print(f"Goals.Model Performance = {e.model_performance()}")
    print(f"Goals.Cost = {e.cost()}")
    print(f"Goals.Network Performance = {e.network_performance()}")
    print(f"Goals.Network Cost = {e.network_cost()}")
    print(f"Goals.Network Fail Probability = {e.network_fail_probability()}")


def generate_times(file_infrastructure, file_latencies):
    total_times = []
    pipelines = [
        #"pipeline_5.yml",
        #"pipeline_10.yml",
        #"pipeline_20.yml",
        "pipeline_40.yml",
        #"pipeline_80.yml",
    ]
    for p in pipelines:
        file_pipeline = f"src/resources/{p}"
        with open(file_pipeline, "r") as input_data_file:
            input_pipeline = input_data_file.read()
        population_size = 60
        pipe_time = []
        # do it 100 times
        for i in range(2):
            start_time = time.time()
            LOGGER.info(f"Executing iteration {i} of {file_pipeline}.")
            Optimizer(
                file_infrastructure=file_infrastructure,
                file_latencies=file_latencies,
                input_pipeline=input_pipeline,
                termination_criterion=StoppingByTotalDominance(idle_evaluations=20),
                population_size=population_size,
            ).run()
            end_time = time.time()
            pipe_time.append(end_time - start_time)
        total_times.append(pipe_time)

    filename = "tmp/times"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerows(total_times)


def generate_pareto(file_infrastructure, file_latencies):
    #file_pipeline = f"src/resources/pipeline_40.yml"
    file_pipeline = f"src/resources/pipeline_80NETXS.yml"
    population_size = 100
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    o = Optimizer(
        file_infrastructure=file_infrastructure,
        file_latencies=file_latencies,
        input_pipeline=input_pipeline,
        # termination_criterion=StoppingByFullPareto(offspring_size=population_size),
        termination_criterion=StoppingByTime(max_seconds=600),
        population_size=population_size,
    )
    o.run()
    pt = ParetoTools(o.get_front())
    pt.save()


def generate_fitnesses(file_infrastructure, file_net_infrastructure, file_latencies):
    #file_pipeline = f"src/resources/pipeline_20.yml"
    file_pipeline = f"src/resources/pipeline_5NET.yml"
    population_size = 50
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    Optimizer(
        file_infrastructure=file_infrastructure,
        file_net_infrastructure=file_net_infrastructure,
        file_latencies=file_latencies,
        input_pipeline=input_pipeline,
        # termination_criterion=StoppingByTotalDominance(idle_evaluations=100),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=population_size * 400
        ),
        observer=WriteObjectivesToFileObserver(),
        population_size=population_size,
    ).run()


def generate_memory(file_infrastructure, file_latencies, number_of_models):
    file_pipeline = f"src/resources/pipeline_{number_of_models}.yml"
    with open(file_pipeline, "r") as input_data_file:
        input_pipeline = input_data_file.read()
    Optimizer(
        file_infrastructure=file_infrastructure,
        file_latencies=file_latencies,
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

    file_infrastructure = "src/resources/infrastructure.csv"
    file_net_infrastructure = "src/resources/network_infrastructure.csv"
    file_latencies = "src/resources/latencies.csv"

    if args.times:
        generate_times(
            file_infrastructure=file_infrastructure, file_latencies=file_latencies
        )

    if args.pareto:
        generate_pareto(
            file_infrastructure=file_infrastructure, file_latencies=file_latencies
        )

    if args.fitnesses:
        generate_fitnesses(
            file_infrastructure=file_infrastructure,
            file_net_infrastructure=file_net_infrastructure,
            file_latencies=file_latencies
        )

    if args.memory:
        generate_memory(
            file_infrastructure=file_infrastructure,
            file_latencies=file_latencies,
            number_of_models=args.memory,
        )

    if args.evaluate:
        evaluate_solution(file_solution=args.evaluate)

    if args.compete:
        compete(
            file_infrastructure=file_infrastructure,
            file_net_infrastructure=file_net_infrastructure,
            file_latencies=file_latencies,
            pipeline=args.compete,
        )


if __name__ == "__main__":
    main()
