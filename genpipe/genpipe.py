#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import random
import yaml
import numpy as np
from numpy.random import choice
import time

LAYERS = [ 'any',
          'premises-tecnalia700',
          'premises-tecnalia101',
          'edge-road',
          'edge-river',
          'cloud-aws',
          'cloud-azure']


class Model:
    def __init__(
        self,
        model_name: str,
        layer: str,
        cpus: int,
        memory: float,
        network: int,
        link: int,
    ):
        self.model = model_name
        self.constraints = {"node": {"layer": layer}}
        self.resources = {"cpus": cpus, "memory": memory, "network": network}
        self.link = link


class PADL:
    def __init__(self, version: str):
        self.version = version
        self.pipeline = []


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )

    text = "This application generates PADL defined analytic models."
    parser = argparse.ArgumentParser(description=text)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-m",
        "--models",
        type=str,
        help="The number of models to be generated.",
        required=True,
    )

    args = parser.parse_args()
    number_of_models = int(args.models)

    total_cpus = 0
    total_memory = 0
    total_network = 0
    total_layers = list()

    p = PADL("1.0")
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))

    for i in range(number_of_models):
        model_size = choice([0, 1, 2], 1, p=[0.8, 0.15, 0.05])[0]
        if model_size == 0:
            # small model
            cpus = random.randint(1, 2)
            memory = round(random.uniform(1, 4), 1)
            network = random.randint(0, 35)
            layer = str(choice(LAYERS, 1, p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])[0])
        elif model_size == 1:
            # medium model
            cpus = random.randint(2, 8)
            memory = round(random.uniform(4, 16), 1)
            network = random.randint(35, 80)
            layer = str(choice(LAYERS, 1, p=[0.6, 0.1, 0.1, 0, 0, 0.1, 0.1])[0])
        elif model_size == 2:
            # large model
            cpus = random.randint(8, 32)
            memory = round(random.uniform(16, 251), 1)
            network = random.randint(80, 200)
            layer = str(choice(LAYERS, 1, p=[0.6, 0.1, 0.1, 0, 0, 0.1, 0.1])[0])

        m = Model(
            model_name=f"m{i}",
            cpus=cpus,
            memory=memory,
            network=network,
            layer=layer,
            link=random.randint(0, number_of_models - 1),
        )
        p.pipeline.append(m.__dict__)

        total_cpus += cpus
        total_memory += memory
        total_network += network
        total_layers.append(layer)

    filename = "tmp/pipeline_{}NET.yml".format(number_of_models)
    print(str(p.pipeline))
    with open(filename, "w", encoding="utf-8") as file:
        file.write("pipeline:\n")
        documents = yaml.dump(p.pipeline, file)
        file.write("version: '" + p.version + "'")
    logging.info(f"{number_of_models} models generated")
    logging.info(f"{total_cpus} total cpus")
    logging.info(f"{total_memory} total memory")
    logging.info(f"{total_network} total network")
    logging.info(f"{total_layers} layers")


if __name__ == "__main__":
    main()
