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
18 % 16
