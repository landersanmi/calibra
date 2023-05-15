import random
import numpy as np
from jmetal.core.operator import Crossover
from jmetal.core.solution import CompositeSolution
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
                if crossover_point_a <= i <= crossover_point_b:
                    temp_p1a = p1a[i]
                    p1a[i] = p2a[i]
                    p2a[i] = temp_p1a

                    temp_p1b = p1b[i]
                    p1b[i] = p2b[i]
                    p2b[i] = temp_p1b

            offspring[0].variables[0].variables = p1a.transpose()
            offspring[0].variables[1].variables = p1b.transpose()
            offspring[1].variables[0].variables = p2a.transpose()
            offspring[1].variables[1].variables = p2b.transpose()

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Power Off CrossOver"


