PIPELINE_COLUMNS = ["cpus", "memory", "network", "link"]

OBJECTIVES_LABELS = ["Model Perf", "Cost", "Net Cost", "Net Fail Prob"]
CONSTRAINT_LABELS = ["cpu", "ram", "deploy", "bandwidth", "net deployment",
                     "net device capacity", "net traffic capacity", "net layers"]
SOLUTION_DF_COLUMNS = OBJECTIVES_LABELS + CONSTRAINT_LABELS
UTOPIAN_CASE = [0.999999, 0.000001, 0.000001, 0.000001]


TEST_PIPELINE_FILENAME = "tests/resources/test_pipeline_{pipeline}.yml"
TEST_INFRASTRUCTURE_FILENAME = "tests/resources/test_infrastructure.csv"
TEST_NETWORK_INFRASTRUCTURE_FILENAME = "tests/resources/test_network_infrastructure.csv"

UNCONSTRAINED_INFRA_FILENAME = "tests/resources/unconstrained_infra.txt"
UNCONSTRAINED_NET_INFRA_FILENAME = "tests/resources/unconstrained_net_infra.txt"