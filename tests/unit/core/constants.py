PIPELINE_COLUMNS = ["cpus", "memory", "network", "link"]

OBJECTIVES_LABELS = ["Model Perf", "Cost", "Net Cost", "Net Fail Prob"]
CONSTRAINT_LABELS = ["cpu", "ram", "deploy", "bandwidth", "net deployment",
                     "net device capacity", "net traffic capacity", "net layers"]
SOLUTION_DF_COLUMNS = OBJECTIVES_LABELS + CONSTRAINT_LABELS
UTOPIAN_CASE = [0.999999, 0.000001, 0.000001, 0.000001]


PIPELINE_FILENAME = "tests/resources/testpipeline_{pipeline}.yml"
INFRASTRUCTURE_FILENAME = "tests/resources/infrastructure.csv"
NETWORK_INFRASTRUCTURE_FILENAME = "tests/resources/network_infrastructure.csv"
