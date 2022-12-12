import pycountry
import pycountry_convert as pc
import yaml
import json
import pandas as pd

from collections import namedtuple

from src.core.constants import CONTINENT_CODES, PIPELINE_COLUMNS


class Parser:
    def __init__(self, input_data: str):
        self.x = self.file2obj(input_data)

    def object_hook(self, d):
        return namedtuple("X", d.keys())(*d.values())

    def file2obj(self, data):
        pass


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
            columns[5]: [],
            columns[6]: [],
        }
        for i in range(len(self.x.pipeline)):
            data[columns[0]].append(float(self.x.pipeline[i].resources.cpus))
            data[columns[1]].append(float(self.x.pipeline[i].resources.memory))
            data[columns[2]].append(float(self.x.pipeline[i].resources.network))
            # get numeric from alpha_2 country code
            country_numeric = pycountry.countries.get(
                alpha_2=self.x.pipeline[i].privacy.location
            ).numeric
            data[columns[3]].append(int(country_numeric))
            data[columns[4]].append(int(self.x.pipeline[i].privacy.type))
            # get continent from alpha_2 country code
            data[columns[5]].append(
                CONTINENT_CODES[
                    pc.country_alpha2_to_continent_code(
                        self.x.pipeline[i].privacy.location
                    )
                ]
            )
            data[columns[6]].append(int(self.x.pipeline[i].link))
        pipeline_df = pd.DataFrame(data=data, columns=columns)
        return pipeline_df