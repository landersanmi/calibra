PIPELINE_COLUMNS = ["cpus", "memory", "network", "layer", "link"]

OBJECTIVES_LABELS = ["Model Perf", "Cost", "Net Cost", "Net Fail Prob"]
CONSTRAINT_LABELS = ["cpu", "ram", "deploy", "bandwidth", "net deployment",
                     "net device capacity", "net traffic capacity", "net layers"]
SOLUTION_DF_COLUMNS = OBJECTIVES_LABELS + CONSTRAINT_LABELS
UTOPIAN_CASE = [-0.999999, 0.000001, 0.000001, 0.000001]
