# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))


from kaggle_shopee.factories.config_factory import ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil

args = ArgsUtil.get_args(EnvEnum.LOCAL, "exp003", [])
config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, False)
data = DataFactory.load_data(config)
data, config = Pp.main(data, config)

# %%
import re

import pandas as pd

pd.set_option("display.max_colwidth", None)
unit = [
    "GR",
    "GM",
    "KG",
    "KILO",
    "MG",
    "LITRE",
    "ML",
    "PC",
    "INCH",
    "YARD",
    "CM",
    "MM",
    "METRE",
    "MICRO",
    "GB",
    "MB",
    "TB",
    "KB",
    "THN",
]

df = pd.read_csv(config.dir_config.input_dir / "valid_df_0.csv")

for u in unit:
    df[f"title_contains_{u}"] = df["title"].str.upper().str.contains(u)
df["title_contains_unit"] = df[[f"title_contains_{u}" for u in unit]].any(axis=1)
