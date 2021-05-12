import torch
import torch.optim
from kaggle_shopee.factories.config_factory import OptimizerConfig


class OptimizerFactory:
    @staticmethod
    def get_optimizer(config: OptimizerConfig, parameters):
        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }
        assert config.name in optimizers, "optimizer_name not in {}".format(
            list(optimizers.keys())
        )
        return optimizers[config.name](parameters, **config.params)
