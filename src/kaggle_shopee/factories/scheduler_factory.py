import torch
import torch.optim
import transformers
from kaggle_shopee.factories.config_factory import SchedulerConfig
from torch.optim import lr_scheduler


class SchedulerFactory:
    @staticmethod
    def get_scheduler(config: SchedulerConfig, optimizer):
        schedulers = {
            "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "get_cosine_schedule_with_warmup": transformers.get_cosine_schedule_with_warmup,
        }
        assert config.name in schedulers, "scheduler_name not in {}".format(
            list(schedulers.keys())
        )
        return schedulers[config.name](optimizer, **config.params)
