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

SOLUTION_DF_COLUMNAMES = ["Model Perf", "Cost", "Network Perf", "Net Cost", "Net Fail Prob",
                          "cpu", "ram", "deploy", "net deployment", "net device capacity", "net traffic capacity", "net layers"]

#OBJECTIVES_LABELS = ['Resilience', 'Model Perf', 'Cost', 'Network Performance', 'Net Cost', 'Net Fail Prob']
OBJECTIVES_LABELS = ['Model Perf', 'Cost', 'Network Performance', 'Net Cost', 'Net Fail Prob']

PIPELINE_FILENAME = "src/resources/pipeline_{pipeline}.yml"
INFRASTRUCTURE_FILENAME = "src/resources/infrastructure.csv"
NETWORK_INFRASTRUCTURE_FILENAME = "src/resources/network_infrastructure.csv"
LATENCIES_FILENAME = "src/resources/latencies.csv"

FITNESSES_FILENAME = "tmp/fitnesses"
PARETO_FILENAME =  "tmp/pareto"
TIMES_FILENAME = "tmp/times"
