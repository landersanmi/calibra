from tensorboardX import SummaryWriter
import datetime

from src.core.constants import OBJECTIVES_LABELS, CONSTRAINT_LABELS

class TensorboardLogger():
    def __init__(self, algo_name="algo", log_dir="logs"):
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(
            self.log_dir + "/" + algo_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def log_objectives(self, objectives, x_axis_value):
        for i, objective in enumerate(objectives):
            label = OBJECTIVES_LABELS[i].replace(" ", "_")
            self.tb_writer.add_scalar("fitnesses/" + label, objective, x_axis_value)

    def log_constraints(self, constraints, constraints_met, x_axis_value):
        self.tb_writer.add_scalar("constraints/constraints_met", constraints_met, x_axis_value)
        for i, constraint in enumerate(constraints):
            label = CONSTRAINT_LABELS[i].replace(" ", "_")
            self.tb_writer.add_scalar("constraints/" + label, constraints[constraint], x_axis_value)
