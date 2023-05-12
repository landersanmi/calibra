import random
import numpy as np
from jmetal.core.operator import Mutation
from jmetal.core.solution import CompositeSolution
from jmetal.util.ckecking import Check


class PowerOffMutation(Mutation[CompositeSolution]):
    def __init__(self, problem, probability: float):
        super(PowerOffMutation, self).__init__(probability=probability)
        self.problem = problem

    def execute(self, solution: CompositeSolution) -> CompositeSolution:
        Check.that(type(solution) is CompositeSolution, "Solution type invalid")
        rand = random.random()

        number_of_objectives = solution.number_of_objectives
        number_of_models = solution.variables[0].number_of_variables
        number_of_devices = len(solution.variables[0].variables[0])
        number_of_net_devices = solution.variables[1].number_of_variables

        #---- PARTE A ----#
        if rand < (1 / (number_of_objectives * 2)):
            self.free_up(number_of_devices, number_of_models, solution)
        else:
            self.bit_flip(number_of_devices, number_of_models, solution)
        self.deployment_mutation(number_of_devices, number_of_models, solution)
        self.irrelevancy_mutation(number_of_devices, number_of_models, solution)
        self.layer_mutation(number_of_devices, solution)

        #---- PARTE B ----#
        self.network_mutation(number_of_devices, number_of_net_devices, solution)

        return solution

    def free_up(self, number_of_devices, number_of_models, solution):
        for i in range(number_of_devices):
            rand = random.random()
            if rand <= self.probability:
                for j in range(number_of_models):
                    rand = random.random()
                    if rand < 0.5:
                        solution.variables[0].variables[j][i] = False

    def bit_flip(self, number_of_devices, number_of_models, solution):
        for i in range(number_of_models):
            for j in range(number_of_devices):
                rand = random.random()
                if rand <= self.probability:
                    solution.variables[0].variables[i][j] = (
                        True if solution.variables[0].variables[i][j] is False else False
                    )

    def deployment_mutation(self, number_of_devices, number_of_models, solution):
        for i in range(number_of_models):
            if sum(solution.variables[0].variables[i]) == 0:
                for j in range(number_of_devices):
                    rand = random.random()
                    if rand <= self.probability/number_of_models:
                        solution.variables[0].variables[i][j] = True

    def irrelevancy_mutation(self, number_of_devices, number_of_models, solution):
        device_sol = np.array(solution.variables[0].variables)

        pipe_cpus = self.problem.pipe.cpus.to_numpy()
        pipe_ram = self.problem.pipe.memory.to_numpy()
        replicated_pipe_cpus = np.tile(pipe_cpus, (device_sol.shape[1], 1))
        replicated_pipe_ram = np.tile(pipe_ram, (device_sol.shape[1], 1))

        cpu_requirements = replicated_pipe_cpus.transpose() * device_sol
        ram_requirements = replicated_pipe_ram.transpose() * device_sol

        cpu_capacities = self.problem.infra.thread_count.to_numpy()
        ram_capacities = self.problem.infra.memory.to_numpy()

        aggregated_cpu_requirements = sum(cpu_requirements)
        aggregated_ram_requirements = sum(ram_requirements)

        device_sol = device_sol.transpose()
        for i in range(number_of_devices):
            if random.random() < self.probability:
                cpu_diff = aggregated_cpu_requirements[i] - cpu_capacities[i]
                ram_diff = aggregated_ram_requirements[i] - ram_capacities[i]
                if cpu_diff > 0 or ram_diff > 0:
                    models_frequencies = sum(device_sol) / number_of_devices
                    models_irrelevancy_cpu = np.zeros(models_frequencies.shape)
                    models_irrelevancy_ram = np.zeros(models_frequencies.shape)
                    if cpu_diff > 0:
                        models_cpu_requirements = cpu_requirements.transpose()[i]
                        models_irrelevancy_cpu = models_frequencies * (models_cpu_requirements / cpu_diff)
                    if ram_diff > 0:
                        models_ram_requirements = ram_requirements.transpose()[i]
                        models_irrelevancy_ram = models_frequencies * (models_ram_requirements / ram_diff)
                    models_irrelevancy = models_irrelevancy_cpu + models_irrelevancy_ram
                    most_irrelevant_models = np.argsort(models_irrelevancy)[::-1][-number_of_models:]
                    for index in most_irrelevant_models:
                        if cpu_diff > 0 and ram_diff > 0:
                            device_sol[i][index] = False
                            cpu_diff -= models_cpu_requirements[index]
                            ram_diff -= models_ram_requirements[index]

        solution.variables[0].variables = device_sol.transpose()

    def network_mutation(self, number_of_devices, number_of_net_devices, solution):
        device_sol = np.array(solution.variables[0].variables).transpose()
        network_sol = np.array(solution.variables[1].variables).transpose()

        net_layers = self.problem.net_infra.layer.to_numpy()
        device_layers = self.problem.infra.cloud_type.to_numpy()
        device_layers[device_layers == 'aws'] = 'cloud'

        for i in range(number_of_devices):
            dev_sum_a = sum(device_sol[i])
            dev_sum_b = sum(network_sol[i])
            if random.random() < self.probability:
                if dev_sum_b == 0 and dev_sum_a != 0:
                    # Activate 1 random net device
                    rand_net_dev = random.randint(0, number_of_net_devices - 1)
                    network_sol[i][rand_net_dev] = True
                elif dev_sum_b == 1 and dev_sum_a < 1:
                    # Deactivate net device
                    network_sol[i] = [False for _ in range(number_of_net_devices)]

                elif dev_sum_b > 1:
                    if dev_sum_a < 1:
                        # Deactivate all net devices
                        network_sol[i] = [False for _ in range(number_of_net_devices)]
                    else:
                        # Hold one random net device and deactivate others
                        deployed = network_sol[i]
                        net_indexes = np.where(deployed == True)
                        rand_net_dev = random.sample(net_indexes, 1)[0][0]
                        network_sol[i] = [False for _ in range(number_of_net_devices)]
                        network_sol[i][rand_net_dev] = True

                if (sum(network_sol[i])) == 1:
                    dev_layer = device_layers[i]
                    net_layer = net_layers[np.where(network_sol[i] == 1)]

                    if dev_layer != net_layer:
                        # Activate a random net device from corresponding layer
                        layer_net_devices_indexes = np.asarray(np.where(net_layers == dev_layer))[0]
                        rand_net_dev = random.choice(layer_net_devices_indexes)
                        network_sol[i] = [False for _ in range(number_of_net_devices)]
                        network_sol[i][rand_net_dev] = True
        solution.variables[1].variables = network_sol.transpose()

    def layer_mutation(self, number_of_devices, solution):
        device_sol = np.array(solution.variables[0].variables).transpose()
        models_layers = self.problem.pipe.layer.to_numpy()

        device_layers = self.problem.infra.cloud_type.to_numpy()
        device_layers[device_layers == 'aws'] = 'cloud'

        for i in range(number_of_devices):
            deployed_model_indexes = np.asarray(np.where(device_sol[i] == True))[0]
            dev_layer = device_layers[i]
            for j in deployed_model_indexes:
                model_layer = models_layers[j]
                if model_layer != "any" and model_layer != dev_layer:
                    layer_devices_indexes = np.asarray(np.where(device_layers == model_layer))[0]
                    rand_dev = random.choice(layer_devices_indexes)
                    device_sol[i][j] = False
                    device_sol[rand_dev][j] = True

        solution.variables[0].variables = device_sol.transpose()

    def get_name(self):
        return "Power Off Mutation"
