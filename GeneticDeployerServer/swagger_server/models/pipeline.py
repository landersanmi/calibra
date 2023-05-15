import yaml
import json
import pandas as pd

from collections import namedtuple

from swagger_server.ai.constants import PIPELINE_COLUMNS


class Parser:
    def __init__(self, input_data: str):
        self.x = self.file2obj(input_data)

    @staticmethod
    def object_hook(d):
        return namedtuple("X", d.keys())(*d.values())


class Pipeline(Parser):
    def file2obj(self, data):
        y = yaml.load(data, Loader=yaml.FullLoader)
        return json.loads(json.dumps(y), object_hook=self.object_hook)

    def load(self) -> pd.DataFrame:
        columns = PIPELINE_COLUMNS
        data = {
            columns[0]: [],
            columns[1]: [],
            columns[2]: [],
            columns[3]: [],
            columns[4]: [],
        }

        for i in range(len(self.x.pipeline)):
            data[columns[0]].append(float(self.x.pipeline[i].resources.cpus))
            data[columns[1]].append(float(self.x.pipeline[i].resources.memory))
            data[columns[2]].append(float(self.x.pipeline[i].resources.network))
            try:
                data[columns[3]].append(str(self.x.pipeline[i].constraints.node.layer))
            except:
                data[columns[3]].append("any")
            data[columns[4]].append(int(self.x.pipeline[i].link))

        pipeline_df = pd.DataFrame(data=data, columns=columns)
        return pipeline_df
