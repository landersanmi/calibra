#!/usr/bin/python3

import logging
import numpy as np
import random

from src.core.utils import (
    Infrastructure,
    NetworkInfrastructure,
    Pipeline,
    ParetoTools,
    Objectives,
    WriteObjectivesToFileObserver,
    Latency,
)
from src.core.utils import (
    StoppingByNonDominance,
    StoppingByTotalDominance,
    StoppingByFullPareto,
    Constraints,
)
from src.core.mutation import PowerOffMutation
from src.core.crossover import PowerOffCrossover

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import (
    NSGAIII,
    UniformReferenceDirectionFactory,
)
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import CompositeSolution, BinarySolution
from jmetal.operator.crossover import CompositeCrossover
from jmetal.lab.visualization import Plot
from jmetal.operator import BitFlipMutation, SPXCrossover, NullCrossover
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, BasicObserver
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    get_non_dominated_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.comparator import StrengthAndKNNDistanceComparator, DominanceComparator


class TravelingModel(BinaryProblem):
    def __init__(self, file_infrastructure, file_network_infrastructure, file_latencies, input_pipeline):
        super(TravelingModel, self).__init__()

        self.infra = Infrastructure(file_infrastructure).load()
        self.net_infra = NetworkInfrastructure(file_network_infrastructure).load()
        self.pipe = Pipeline(input_pipeline).load()
        self.ld = Latency(file_location=file_latencies).load()

        # number of models
        # self.number_of_models = self.pipe.shape[1]
        self.number_of_models = self.pipe.shape[0]
        self.number_of_objectives = 5
        # number of devices
        self.number_of_devices = len(self.infra.index)
        # number of network devices
        self.number_of_net_devices = len(self.net_infra.index)
        # number of constraints
        self.number_of_constraints = 7

        # self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]

        #self.obj_labels = ['Resilience', 'Model Perf', 'Cost', 'Network Performance', 'Net Cost', 'Net Fail Prob']
        self.obj_labels = ['Model Perf', 'Cost', 'Network Performance', 'Net Cost', 'Net Fail Prob']
        #self.obj_labels = ['Net Cost', 'Net Fail Prob']

        # self.obj_directions = [self.MAXIMIZE]
        # self.obj_labels = ['performance']

    def evaluate(self, solution: CompositeSolution) -> CompositeSolution:
        self.objectives = Objectives()

        '''
        solution.objectives[0] = -1 * self.objectives.get_resilience(
            self.infra, solution
        )
        '''
        solution.objectives[0] = -1 * self.objectives.get_performance(
            self.pipe, self.infra, solution
        )
        solution.objectives[1] = self.objectives.get_consumption(
            self.pipe, self.infra, solution
        )
        solution.objectives[2] = -1 * self.objectives.get_network_performance(
            ld=self.ld, pipe=self.pipe, infra=self.infra, solution=solution
        )
        solution.objectives[3] = self.objectives.get_net_cost(
            net_infra=self.net_infra, solution=solution
        )
        solution.objectives[4] = self.objectives.get_net_fail_probability(
            net_infra=self.net_infra, solution=solution
        )
        '''
        solution.objectives[0] = self.objectives.get_net_cost(
            net_infra=self.net_infra, solution=solution
        )
        solution.objectives[1] = self.objectives.get_net_fail_probability(
            net_infra=self.net_infra, solution=solution
        )
        '''

        self.__evaluate_constraints(solution)
        return solution

    def __evaluate_constraints(self, solution: CompositeSolution) -> None:
        # constraints = [0.0 for _ in range(self.number_of_constraints)]
        constraints = []

        c = Constraints(solution, self.infra, self.net_infra, self.pipe)
        """ 
        do not exceed total CPU per device
        """
        # x = s.transpose() * self.pipe.cpus.to_numpy()
        # sum_rows = np.sum(x, axis=1)
        # thread_count = self.infra.thread_count.to_numpy()
        # constraints.append(0 if not (thread_count < sum_rows).any() else -1)
        constraints.append(c.cpu_constraint())

        """
        do not exceed total RAM per device
        """
        # x = s.transpose() * self.pipe.memory.to_numpy()
        # sum_rows = np.sum(x, axis=1)
        # memory = self.infra.memory.to_numpy()
        # constraints.append(0 if not (memory < sum_rows).any() else -1)
        constraints.append(c.ram_constraint())

        """ 
        each model should be deployed in at least one device
        """
        # sum_rows = np.sum(s, axis=1)
        # the sum of all rows should be bigger or equal to one
        # constraints.append(0 if not (sum_rows < 1).any() else -1)
        constraints.append(c.deployment_constraint())

        """
        do not exceed total GPU per device
        """

        """ 
        enforce privacy constraints
        """
        #constraints.append(c.privacy_constraint())

        """
        each device with at least one model deployed 
        should be connected to one network device.
        """
        constraints.append(c.net_deployment_constraint())

        """
        each net device maximum users capacity must be complained
        """
        constraints.append(c.net_device_capacity_constraint())

        """
        each net device maximum traffic capacity must be complained
        """
        constraints.append(c.net_traffic_capacity_constraint())

        """
        each device must be related to a net device of the same layer
        """
        constraints.append(c.net_layers_constraint())

        solution.constraints = constraints

    def create_solution(self) -> CompositeSolution:
        model_solution = BinarySolution(number_of_objectives=self.number_of_objectives,
                                        number_of_constraints=self.number_of_constraints,
                                        number_of_variables=self.number_of_models)
        network_solution = BinarySolution(number_of_objectives=self.number_of_objectives,
                                        number_of_constraints=self.number_of_constraints,
                                        number_of_variables=self.number_of_net_devices)

        for i in range(self.number_of_models):
            model_solution.variables[i] = [
               True if random.random() > 0.5 else False
               for _ in range(self.number_of_devices)
            ]

        net_sol = list()
        for i in range(self.number_of_devices):
            rand_net = random.randint(0, self.number_of_net_devices)
            net_sol.append([
               True if j == rand_net else False for j in range(self.number_of_net_devices)
            ])
        net_sol = np.array(net_sol)
        network_solution.variables = net_sol.transpose()
        new_solution = CompositeSolution([model_solution, network_solution])
        return new_solution

    def get_name(self) -> str:
        return "TravelingModel"


class Optimizer:
    def __init__(
        self,
        termination_criterion,
        file_infrastructure,
        file_net_infrastructure,
        file_latencies,
        input_pipeline,
        population_size=100,
        observer=None,
    ):
        self.termination_criterion = termination_criterion
        self.file_infrastructure = file_infrastructure
        self.file_net_infrastructure = file_net_infrastructure
        self.file_latencies = file_latencies
        self.input_pipeline = input_pipeline
        self.observer = observer
        self.population_size = population_size

    def run(self):
        self.problem = TravelingModel(
            file_infrastructure=self.file_infrastructure,
            file_network_infrastructure=self.file_net_infrastructure,
            file_latencies=self.file_latencies,
            input_pipeline=self.input_pipeline,
        )

        self.algorithm = NSGAIII(
            problem=self.problem,
            population_size=self.population_size,
            # offspring_population_size=self.population_size,
            # reference_directions=UniformReferenceDirectionFactory(4, n_points=92),
            reference_directions=UniformReferenceDirectionFactory(self.problem.number_of_objectives, n_points=92),
            # mutation=BitFlipMutation(probability=1.0 / self.problem.number_of_devices),
            #mutation=PowerOffMutation(probability=1.0 / self.problem.number_of_devices),
            mutation=PowerOffMutation(problem=self.problem, probability=1.0),
            crossover=PowerOffCrossover(probability=1.0),
            termination_criterion=self.termination_criterion,
            # termination_criterion=StoppingByEvaluations(max_evaluations=self.max_evaluations),
            # termination_criterion=StoppingByNonDominance(idle_evaluations=200),
            # termination_criterion=StoppingByTotalDominance(idle_evaluations=50),
            # termination_criterion=StoppingByFullPareto(self.population_size),
            # dominance_comparator=StrengthAndKNNDistanceComparator()
            dominance_comparator=DominanceComparator(),
        )

        # if (self.interactive_plot):
        #    self.algorithm.observable.register(observer=ProgressBarObserver(max=self.max_evaluations))
        #    self.algorithm.observable.register(observer=VisualizerObserver(reference_front=self.problem.reference_front, display_frequency=100))
        #    basic = BasicObserver(frequency=1.0)
        #    self.algorithm.observable.register(observer=basic)

        if self.observer:
            self.algorithm.observable.register(observer=self.observer)

        self.algorithm.run()

        # self.front = get_non_dominated_solutions(self.algorithm.get_result())
        self.front = self.algorithm.get_result()
        # logging.warning(self.front[0].objectives)
        # logging.warning(self.front[1].objectives)
        # logging.warning(self.front[2].objectives)
        # logging.warning(self.front[3].objectives)

        # for v in self.front[0].variables:
        #    logging.warning(v)
        # logging.warning(self.front[0].objectives)
        # logging.info(f'Algorithm: ${self.algorithm.get_name()}')
        # logging.info(f'Problem: ${self.problem.get_name()}')
        logging.info(f"Computing time: ${self.algorithm.total_computing_time}")

    def get_front(self):
        return self.front

    def plot(self):
        # Plot front
        plot_front = Plot(
            title="Pareto front approximation. Problem: " + self.problem.get_name(),
            reference_front=self.problem.reference_front,
            axis_labels=self.problem.obj_labels,
        )
        plot_front.plot(
            self.front, label=self.algorithm.label, filename=self.algorithm.get_name()
        )

        # print variables and fitnesses
        print_function_values_to_file(self.front, "FUN." + self.algorithm.label)
        print_variables_to_file(self.front, "VAR." + self.algorithm.label)
