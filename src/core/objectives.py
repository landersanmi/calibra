import numpy as np

from src.core.models.pipeline import Pipeline
from src.core.models.infrastructure import Infrastructure
from src.core.models.network_infrastructure import NetworkInfrastructure

from jmetal.core.solution import CompositeSolution


class Objectives:
    def get_consumption(
        self, pipe: Pipeline, infra: Infrastructure, solution: CompositeSolution
    ) -> int:
        solution = solution.variables[0]
        s = np.asfarray(solution.variables, dtype=bool)

        threads_required = s.transpose().dot(pipe.cpus.to_numpy()).astype(int)

        consumption = [
            m[min(n, len(m) - 1)] for m, n in zip(infra.consumption, threads_required)
        ]

        _, number_of_devices = s.shape
        return sum(consumption) / number_of_devices

    def get_performance(
        self, pipe: Pipeline, infra: Infrastructure, solution: CompositeSolution
    ) -> int:
        solution = solution.variables[0]
        s = np.asfarray(solution.variables, dtype=bool)

        base_performance = s * infra.performance.to_numpy()

        threads_required = s.transpose().dot(pipe.cpus.to_numpy()).astype(int)

        coefficient = [
            m[min(n - 1, len(m) - 1)]
            for m, n in zip(infra.parallelization, threads_required)
        ]

        number_of_models, _ = s.shape

        x = base_performance * coefficient
        x = np.ma.masked_array(x, mask=s == 0)
        return np.nanmean(x, axis=1).sum() / number_of_models

    def get_net_cost(
            self,
            net_infra: NetworkInfrastructure,
            solution: CompositeSolution,
    ) -> int:
        solution = np.asfarray(solution.variables[1].variables, dtype=bool)
        deployed_net_devices = np.sum(solution, axis=1)
        deployed_net_devices = np.where(deployed_net_devices >= 1, 1, 0)
        cost_per_net_device = net_infra.cost.to_numpy()
        net_cost = np.sum(deployed_net_devices * cost_per_net_device)
        return net_cost

    def get_net_fail_probability(
            self,
            net_infra: NetworkInfrastructure,
            solution: CompositeSolution,
    ) -> int:
        solution = np.asfarray(solution.variables[1].variables, dtype=bool)
        deployed_net_devices = np.sum(solution, axis=1)
        deployed_net_devices = np.where(deployed_net_devices >= 1, 1, 0)
        failure_per_net_device = net_infra.failure_prob.to_numpy()
        net_fail_probability = np.sum(deployed_net_devices * failure_per_net_device)
        return net_fail_probability