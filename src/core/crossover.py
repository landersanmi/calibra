import random

import numpy as np
from jmetal.core.operator import Crossover
from jmetal.core.solution import BinarySolution, CompositeSolution
from jmetal.util.ckecking import Check
from typing import List
import copy


class PowerOffCrossover(Crossover[CompositeSolution, CompositeSolution]):
    def __init__(self, probability: float):
        super(PowerOffCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[CompositeSolution]) -> List[CompositeSolution]:
        Check.that(type(parents[0]) is CompositeSolution, "Solution type invalid")
        Check.that(type(parents[1]) is CompositeSolution, "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents)
        rand = random.random()

        p1a = np.array(parents[0].variables[0].variables).transpose()
        p1b = np.array(parents[0].variables[1].variables).transpose()
        p2a = np.array(parents[1].variables[0].variables).transpose()
        p2b = np.array(parents[1].variables[1].variables).transpose()
        if rand < self.probability:
            total_devices = len(parents[0].variables[0].variables[0])
            crossover_point_a = random.randrange(0, total_devices-1)
            crossover_point_b = random.randrange(crossover_point_a, total_devices)
            for i in range(total_devices):
                if i >= crossover_point_a and i <= crossover_point_b:
                    tempp1a = p1a[i]
                    p1a[i] = p2a[i]
                    p2a[i] = tempp1a

                    tempp1b = p1b[i]
                    p1b[i] = p2b[i]
                    p2b[i] = tempp1b

            offspring[0].variables[0].variables = p1a.transpose()
            offspring[0].variables[1].variables = p1b.transpose()
            offspring[1].variables[0].variables = p2a.transpose()
            offspring[1].variables[1].variables = p2b.transpose()

        return offspring
    """
    def execute(self, parents: List[CompositeSolution]) -> List[CompositeSolution]:
        Check.that(type(parents[0]) is CompositeSolution, "Solution type invalid")
        Check.that(type(parents[1]) is CompositeSolution, "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents)
        rand = random.random()

        if rand <= self.probability:
            # 1. Get the total number of bits
            total_number_of_bits = parents[0].variables[0].get_total_number_of_bits()

            # 2. Calculate the point to make the crossover
            crossover_point = random.randrange(0, total_number_of_bits)

            # 3. Compute the variable containing the crossover bit
            variable_to_cut = 0
            bits_count = len(parents[1].variables[0].variables[variable_to_cut])
            while bits_count < (crossover_point + 1):
                variable_to_cut += 1
                bits_count += len(parents[1].variables[0].variables[variable_to_cut])

            # 4. Compute the bit into the selected variable
            diff = bits_count - crossover_point
            crossover_point_in_variable = len(parents[1].variables[0].variables[variable_to_cut]) - diff

            # 5. Apply the crossover to the variable
            bitset1 = copy.copy(parents[0].variables[0].variables[variable_to_cut])
            bitset2 = copy.copy(parents[1].variables[0].variables[variable_to_cut])

            for i in range(crossover_point_in_variable, len(bitset1)):
                swap = bitset1[i]
                bitset1[i] = bitset2[i]
                bitset2[i] = swap

            offspring[0].variables[0].variables[variable_to_cut] = bitset1
            offspring[1].variables[0].variables[variable_to_cut] = bitset2

            # 6. Apply the crossover to the other variables
            for i in range(variable_to_cut + 1, parents[0].variables[0].number_of_variables):
                offspring[0].variables[0].variables[i] = copy.deepcopy(parents[1].variables[0].variables[i])
                offspring[1].variables[0].variables[i] = copy.deepcopy(parents[0].variables[0].variables[i])

        return offspring
    """
    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Power Off CrossOver"


