CONTINENT_CODES = {
    "AF": 100,
    "AN": 101,
    "AS": 102,
    "EU": 103,
    "NA": 104,
    "OC": 105,
    "SA": 106,
}

PIPELINE_COLUMNS = ["cpus", "memory", "network", "country", "privacy_type", "continent", "link"]

OBJECTIVES_LABELS = ["Model Perf", "Cost", "Net Cost", "Net Fail Prob"]
CONSTRAINT_LABELS = ["cpu", "ram", "deploy", "bandwidth", "net deployment", "net device capacity", "net traffic capacity", "net layers"]
SOLUTION_DF_COLUMNAMES = OBJECTIVES_LABELS + CONSTRAINT_LABELS


PIPELINE_FILENAME = "src/resources/pipeline_{pipeline}.yml"
INFRASTRUCTURE_FILENAME = "src/resources/infrastructure.csv"
NETWORK_INFRASTRUCTURE_FILENAME = "src/resources/network_infrastructure.csv"

FITNESSES_FILENAME = "tmp/data/fitnesses/fitnesses.txt"
PARETO_FILENAME = "tmp/data/paretos/pareto.txt"
TIMES_FILENAME = "tmp/data/times/times.txt"
