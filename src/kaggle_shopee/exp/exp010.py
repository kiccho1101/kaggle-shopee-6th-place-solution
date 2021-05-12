# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))


from kaggle_shopee.factories.config_factory import ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil

args = ArgsUtil.get_args(EnvEnum.LOCAL, "exp017", [])
config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, False)
data = DataFactory.load_data(config)
data, config = Pp.main(data, config)


# %%
import pycld2
import texthero as hero

data.train["title_cleaned"] = hero.clean(data.train["title"])

data.train["title_lang"] = (
    data.train["title"].fillna("").map(lambda x: pycld2.detect(x)[2][0][1])
)

# %%
data.train[["title", "title_cleaned"]].sample(10)

# %%
data.train[~data.train["title"].map(lambda x: "\\" in x)].sample(10)[
    ["title", "title_cleaned"]
]
