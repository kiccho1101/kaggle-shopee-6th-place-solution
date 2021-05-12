from typing import List

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from kaggle_shopee.factories.config_factory import TransformComponent


def get_transform(config: TransformComponent):
    return getattr(A, config.name)(**config.args)


class TransformFactory:
    @staticmethod
    def get_transforms(configs: List[TransformComponent]) -> A.Compose:
        return A.Compose(
            [get_transform(config) for config in configs] + [ToTensorV2(p=1.0)],
            p=1.0,
        )
