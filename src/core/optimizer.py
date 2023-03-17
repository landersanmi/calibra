import logging

from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.lab.visualization import Plot
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, BasicObserver
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    get_non_dominated_solutions,
)
from jmetal.util.comparator import StrengthAndKNNDistanceComparator, DominanceComparator

from src.core.mutation import PowerOffMutation
from src.core.crossover import PowerOffCrossover
from src.core.problem import TravelingModel

class Optimizer:
    def __init__(
        self,
        termination_criterion,
        #dominance_comparator,
        file_infrastructure,
        file_net_infrastructure,
        input_pipeline,
        population_size=100,
        observer=None,
    ):
        self.termination_criterion = termination_criterion
        #self.dominance_comparator = dominance_comparator
        self.file_infrastructure = file_infrastructure
        self.file_net_infrastructure = file_net_infrastructure
        self.input_pipeline = input_pipeline
        self.observer = observer
        self.population_size = population_size
        self.problem = None
        self.algorithm = None
        self.front = None

    def run(self):
        self.problem = TravelingModel(
            file_infrastructure=self.file_infrastructure,
            file_network_infrastructure=self.file_net_infrastructure,
            input_pipeline=self.input_pipeline,
        )

        self.algorithm = NSGAIII(
            problem=self.problem,
            population_size=self.population_size,
            # offspring_population_size=self.population_size,
            reference_directions=UniformReferenceDirectionFactory(self.problem.number_of_objectives, n_points=92),
            crossover=PowerOffCrossover(probability=1.0),
            mutation=PowerOffMutation(problem=self.problem, probability=1.0),
            termination_criterion=self.termination_criterion,
            #dominance_comparator=self.dominance_comparator,
        )

        #if (self.interactive_plot):
            #self.algorithm.observable.register(observer=ProgressBarObserver(max=self.max_evaluations))
            #self.algorithm.observable.register(observer=VisualizerObserver(reference_front=self.problem.reference_front, display_frequency=100))
            #basic = BasicObserver(frequency=1.0)
            #self.algorithm.observable.register(observer=basic)

        if self.observer:
            self.algorithm.observable.register(observer=self.observer)

        self.algorithm.run()

        # self.front = get_non_dominated_solutions(self.algorithm.get_result())
        self.front = self.algorithm.get_result()

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
