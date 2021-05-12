# %%
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from enum import Enum
from typing import Dict, List, Tuple, Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
import torch.cuda.amp
import torch.nn as nn
import torch.nn.functional as F
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import Config, ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.metric_learning_factory import MetricLearningFactory
from kaggle_shopee.factories.optimizer_factory import OptimizerFactory
from kaggle_shopee.factories.scheduler_factory import SchedulerFactory
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.file_util import FileUtil
from kaggle_shopee.utils.global_util import GlobalUtil
from kaggle_shopee.utils.img_util import ImgUtil
from kaggle_shopee.utils.metric_util import MetricUtil
from kaggle_shopee.utils.mlflow_util import MlflowUtil
from kaggle_shopee.utils.test_util import TestUtil
from kaggle_shopee.utils.time_util import TimeUtil
from numpy.lib import math
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm


class Pp:
    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def image_path(data: Data, config: Config) -> Tuple[Data, Config]:
        data.train["image_path"] = data.train["image"].map(
            lambda i: str(config.dir_config.train_images_dir / i)
        )
        data.test["image_path"] = data.test["image"].map(
            lambda i: str(config.dir_config.test_images_dir / i)
        )
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def split_folds(data: Data, config: Config) -> Tuple[Data, Config]:
        folds = GroupKFold(n_splits=config.cv_config.n_splits)
        data.train["fold"] = -1
        for fold, (train_idx, valid_idx) in enumerate(
            folds.split(data.train, None, data.train[config.cv_config.target])
        ):
            data.train.loc[valid_idx, "fold"] = fold
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def label_group_le(data: Data, config: Config) -> Tuple[Data, Config]:
        le = LabelEncoder()
        data.train["label_group_le"] = le.fit_transform(data.train["label_group"])
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def target(data: Data, config: Config) -> Tuple[Data, Config]:
        tmp = data.train.groupby("label_group")["posting_id"].unique()
        data.train["target"] = data.train["label_group"].map(tmp)
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    def main(data: Data, config: Config) -> Tuple[Data, Config]:
        data, config = Pp.image_path(data, config)
        data, config = Pp.split_folds(data, config)
        data, config = Pp.label_group_le(data, config)
        data, config = Pp.target(data, config)
        return data, config


class Fe:
    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def image_phash_match_posting_ids(
        data: Data, config: Config
    ) -> Tuple[Data, Config]:
        _map = data.train.groupby("image_phash")["posting_id"].unique()
        data.train["image_phash_match_posting_ids"] = data.train["image_phash"].map(
            _map
        )
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    def main(data: Data, config: Config) -> Tuple[Data, Config]:
        data, config = Fe.image_phash_match_posting_ids(data, config)
        return data, config


args = ArgsUtil.get_args(env=EnvEnum.LOCAL, exp="exp003")
print(args)

config = ConfigFactory.get_config_from_yaml_file(args.exp)
GlobalUtil.seed_everything(config.seed)
data = DataFactory.load_data(config)
if len(data.test) > 3:
    config.is_submitting = True
data, config = Pp.main(data, config)
# data, config = Fe.main(data, config)

# %%
data.sample_submission


# %%

MlflowUtil.start_run(config.mlflow_config, config.exp, config.name, "exp", True)
MlflowUtil.log_params_config(config)
for fold in range(config.cv_config.n_splits):
    train_dataloader, valid_dataloader = DataLoaderFactory.get_cv_dataloaders(
        data, fold, config
    )
    checkpoint_path = (
        f"{args.exp}_{fold}_{config.model_config.model_arch}"
        + "_{epoch:02d}_{valid_f1:.4f}"
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=str(config.dir_config.checkpoint_dir),
        filename=checkpoint_path,
        save_top_k=1,
        monitor="valid_f1",
        mode="max",
        verbose=True,
    )
    model = lit_models.ShopeeLitModel(data, config, fold)
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=config.train_config.epochs,
        checkpoint_callback=checkpoint_callback,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=1,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
MlflowUtil.end_run()
