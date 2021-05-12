# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))


import numpy as np
from kaggle_shopee.factories.config_factory import ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.global_util import GlobalUtil

args = ArgsUtil.get_args(EnvEnum.LOCAL, "exp003", [])
config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, False)
GlobalUtil.seed_everything(config.seed)
data = DataFactory.load_data(config)
data, config = Pp.main(data, config)

# %%
import matplotlib.pyplot as plt

n_sqrt = 2
a = 0.2
b = 0.3

train_df = data.train[data.train["fold"] == 0].reset_index(drop=True).copy()
label_counts = train_df["label_group_le"].value_counts().to_dict()
tmp = train_df["label_group_le"].value_counts().to_dict()

for _ in range(n_sqrt):
    tmp = {k: np.sqrt(v) for k, v in tmp.items()}
tmp = {k: 1 / v for k, v in tmp.items()}

_min = np.min(list(tmp.values()))
_max = np.max(list(tmp.values()))

tmp = {k: (v - _min) / (_max - _min) for k, v in tmp.items()}
tmp = {k: a * v + b for k, v in tmp.items()}
tmp = {k: a + b - v + b for k, v in tmp.items()}


labels = list(label_counts.keys())
x = [label_counts[_label] for _label in labels]
y = [tmp[_label] for _label in labels]
plt.scatter(x, y)

# %%
train_df["label_group_le"].value_counts().value_counts()
