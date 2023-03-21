import numpy as np


class Constraints:

    def __init__(self, solution, infra, net_infra,  pipe):
        self.model_solution = np.asfarray(solution.variables[0].variables, dtype=bool)
        self.network_solution = np.asfarray(solution.variables[1].variables, dtype=bool)
        self.infra = infra
        self.net_infra = net_infra
        self.pipe = pipe

    def cpu_constraint(self):
        x = self.model_solution.transpose() * self.pipe.cpus.to_numpy()
        sum_rows = np.sum(x, axis=1)
        thread_count = self.infra.thread_count.to_numpy()
        c = [i if i > 1 else 0 for i in sum_rows / thread_count]
        return 0 if not (thread_count <= sum_rows).any() else -1 * np.sum(c)

    def deployment_constraint(self):
        sum_rows = np.sum(self.model_solution, axis=1)
        # the sum of all rows should be bigger or equal to one
        return 0 if not (sum_rows < 1).any() else -1 * np.sum(sum_rows < 1)

    def ram_constraint(self):
        x = self.model_solution.transpose() * self.pipe.memory.to_numpy()
        sum_rows = np.sum(x, axis=1)
        memory = self.infra.memory.to_numpy()
        c = [i if i > 0 else 0 for i in sum_rows / memory]
        return 0 if not (memory < sum_rows).any() else -1 * np.sum(c)

    def bandwidth_constraint(self):
        models_net_requirements = self.pipe.network.to_numpy()
        models_net_requirements = np.tile(models_net_requirements, (self.model_solution.shape[1], 1))
        models_net_requirements = models_net_requirements*self.model_solution.transpose()

        devices_net_requirements = np.sum(models_net_requirements, axis=1)
        bandwidth_per_device = self.infra.bandwidth.to_numpy()

        result = 0
        for required, bandwidth in zip(devices_net_requirements, bandwidth_per_device):
            if required > bandwidth:
                result -= (required - bandwidth)
        return result

    def net_deployment_constraint(self):
        models_per_device = np.sum(self.model_solution.transpose(), axis=1)
        deployed_devices = np.where(models_per_device >= 1, 1, 0)
        net_devices_per_device = np.sum(self.network_solution.transpose(), axis=1)
        result = 0
        for i, j in zip(deployed_devices, net_devices_per_device):
            if i != j:
                result -= abs(j-i)
        return result

    def net_device_capacity_constraint(self):
        devices_per_net_device = np.sum(self.network_solution, axis=1)
        net_devices_capacities = self.net_infra.max_devices.to_numpy()
        result = 0
        for i, dev in enumerate(devices_per_net_device):
            if dev > net_devices_capacities[i]:
                result -= (dev - net_devices_capacities[i])
        return result

    def net_traffic_capacity_constraint(self):
        models_net_requirements = self.pipe.network.to_numpy()
        models_net_requirements = np.tile(models_net_requirements, (self.model_solution.shape[1], 1))
        models_net_requirements = models_net_requirements*self.model_solution.transpose()

        devices_net_requirements = np.sum(models_net_requirements, axis=1)
        devices_net_requirements = np.tile(devices_net_requirements, (self.network_solution.shape[0], 1))

        total_traffic_per_net_device = devices_net_requirements*self.network_solution
        requirements_per_net = np.sum(total_traffic_per_net_device, axis=1)
        max_traffic_per_net = self.net_infra.max_network_traffic.to_numpy()

        result = 0
        for required, max_traffic in zip(requirements_per_net, max_traffic_per_net):
            if required > max_traffic:
                result -= (required - max_traffic)
        return result

    def net_layers_constraint(self):
        models_layers = self.pipe.layer.to_numpy()
        net_layers = self.net_infra.layer.to_numpy()
        device_layers = self.infra.cloud_type.to_numpy()
        device_layers[device_layers == 'aws'] = 'cloud'
        #device_layers[device_layers == 'premises'] = 'cloud'

        result = 0
        for i, row in enumerate(self.model_solution):
            for j, deployed in enumerate(row):
                if deployed and models_layers[i] != "any" and device_layers[j] != models_layers[i]:
                    result -= 1
        for i, row in enumerate(self.network_solution):
            for j, deployed in enumerate(row):
                if deployed and device_layers[j] != net_layers[i]:
                    result -= 1
        return result
