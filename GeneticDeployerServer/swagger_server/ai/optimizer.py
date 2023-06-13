import logging
import pandas as pd
import numpy as np
from scipy import spatial

from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.lab.visualization import Plot
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, BasicObserver
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    get_non_dominated_solutions,
)

from swagger_server.ai.mutation import PowerOffMutation
from swagger_server.ai.crossover import PowerOffCrossover
from swagger_server.ai.problem import DeploymentProblem
from swagger_server.ai.constants import (
    SOLUTION_DF_COLUMNS,
    OBJECTIVES_LABELS,
    UTOPIAN_CASE
)


class Optimizer:
    def __init__(
        self,
        termination_criterion,
        file_infrastructure,
        file_net_infrastructure,
        input_pipeline,
        population_size=100,
        observer=None,
    ):
        self.termination_criterion = termination_criterion
        self.file_infrastructure = file_infrastructure
        self.file_net_infrastructure = file_net_infrastructure
        self.input_pipeline = input_pipeline
        self.observer = observer
        self.population_size = population_size
        self.problem = None
        self.algorithm = None
        self.front = None

    def run(self):
        self.problem = DeploymentProblem(
            file_infrastructure=self.file_infrastructure,
            file_network_infrastructure=self.file_net_infrastructure,
            input_pipeline=self.input_pipeline,
        )

        self.algorithm = NSGAIII(
            problem=self.problem,
            population_size=self.population_size,
            reference_directions=UniformReferenceDirectionFactory(self.problem.number_of_objectives, n_points=92),
            crossover=PowerOffCrossover(probability=1.0),
            mutation=PowerOffMutation(problem=self.problem, probability=1.0),
            termination_criterion=self.termination_criterion,
        )

        #if (self.interactive_plot):
            #self.algorithm.observable.register(observer=ProgressBarObserver(max=self.max_evaluations))
            #self.algorithm.observable.register(observer=VisualizerObserver(reference_front=self.problem.reference_front, display_frequency=100))
            #basic = BasicObserver(frequency=1.0)
            #self.algorithm.observable.register(observer=basic)

        if self.observer:
            self.algorithm.observable.register(observer=self.observer)

        self.algorithm.run()
        self.front = self.algorithm.get_result()

        logging.info(f"Computing time: ${self.algorithm.total_computing_time}")

    def get_front(self):
        return self.front

    def get_best_solution(self, include_fitness_specific=False):
        objectives_and_constraints, objectives = [], []
        for s in self.get_front():
            objectives_and_constraints.append(s.objectives + s.constraints)
            objectives.append(s.objectives)

        df = pd.DataFrame(objectives_and_constraints, columns=SOLUTION_DF_COLUMNS)
        objectives_df = pd.DataFrame(objectives, columns=OBJECTIVES_LABELS)

        row_dict = dict()
        for objective, utopian_value in zip(OBJECTIVES_LABELS, UTOPIAN_CASE):
            row_dict[objective] = utopian_value
        row = pd.Series(row_dict)
        objectives_df = pd.concat([objectives_df, row.to_frame().T], ignore_index=True)

        objectives_df['cosine_distance'] = objectives_df.apply(lambda row: spatial.distance.cosine(row.values, objectives_df.iloc[-1]), axis=1)
        objectives_df = objectives_df.sort_values(by=['cosine_distance'], ascending=True)

        best_solutions = [objectives_df.head(2)[1:2].index]

        if include_fitness_specific:
            for i, col in enumerate(OBJECTIVES_LABELS):
                if i < self.problem.number_of_objectives:
                    print("--------Goals."+col+"---------")
                    print(df.sort_values(by=[col]).head(1))
                    best_solutions = np.append(best_solutions, df.sort_values(by=[col]).head(1).index)
        else:
            return best_solutions[0]

        return best_solutions

    def get_best_solution_values(self, best_solutions):
        solutions = []
        print(best_solutions)
        front = self.get_front()
        for idx in best_solutions:
            solutions.append(front[idx])
        return solutions
