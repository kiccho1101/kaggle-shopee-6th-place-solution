# %%
import sys
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

sys.path.append(str(Path.cwd().parents[1]))

import numpy as np
import pandas as pd
from fastai.vision.all import (AdaptiveAvgPool, ConvLayer, Flatten,
                               ImageDataLoaders, Learner, URLs, accuracy,
                               untar_data)
from kaggle_shopee.factories.config_factory import (Config, ConfigFactory,
                                                    EnvEnum, ReducerEnum)
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.file_util import FileUtil
from kaggle_shopee.utils.global_util import GlobalUtil
from kaggle_shopee.utils.img_util import ImgUtil
from kaggle_shopee.utils.metric_util import MetricUtil
from kaggle_shopee.utils.mlflow_util import MlflowUtil
from kaggle_shopee.utils.test_util import TestUtil
from kaggle_shopee.utils.time_util import TimeUtil
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
    def target(data: Data, config: Config) -> Tuple[Data, Config]:
        _map = data.train.groupby("label_group")["posting_id"].unique()
        data.train["target"] = data.train["label_group"].map(_map)
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    def main(data: Data, config: Config) -> Tuple[Data, Config]:
        data, config = Pp.image_path(data, config)
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


args = ArgsUtil.get_args(env=EnvEnum.LOCAL, exp="exp002")

config = ConfigFactory.get_config_from_yaml_file(args.exp)
data = DataFactory.load_data(config)
if len(data.test) > 3:
    config.is_submitting = True
data, config = Pp.main(data, config)
data, config = Fe.main(data, config)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

emb_size = 3
output_classes = 10
batch_size = 1

classifier = nn.Linear(emb_size, output_classes, bias=False)

W = classifier.weight.T
x = torch.rand((batch_size, emb_size))

x = F.normalize(x)
W = F.normalize(W, dim=0)


# %%


def arcface_loss(cosine, targ, m=0.4):
    # this prevents nan when a value slightly crosses 1.0 due to numerical error
    cosine = cosine.clip(-1 + 1e-7, 1 - 1e-7)
    # Step 3:
    arcosine = cosine.arccos()
    # Step 4:
    arcosine += F.one_hot(targ, num_classes=output_classes) * m
    # Step 5:
    cosine2 = arcosine.cos()
    # Step 6:
    return F.cross_entropy(cosine2, targ)


class ArcFaceClassifier(nn.Module):
    def __init__(self, emb_size: int, output_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_size, output_classes))
        torch.nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        return x_norm @ W_norm


class ArcFaceLoss(nn.Module):
    def __init__(self, output_classes: int, m: float = 0.4):
        super().__init__()
        self.m = m
        self.output_classes = output_classes

    def forward(self, cosine, target):
        cosine = cosine.clip(-1 + 1e-7, 1 - 1e-7)
        arccos = cosine.arccos()
        arccos += F.one_hot(target, num_classes=self.output_classes) * self.m
        cos = arccos.cos()
        return F.cross_entropy(cos, target)


class SimpleConv(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        ch_in = [3, 6, 12, 24]
        convs = [ConvLayer(c, c * 2, stride=2) for c in ch_in]
        convs += [AdaptiveAvgPool(), Flatten(), nn.Linear(48, emb_size)]
        self.convs = nn.Sequential(*convs)
        self.classifier = classifier

    def get_embs(self, x):
        return self.convs(x)

    def forward(self, x):
        x = self.get_embs(x)
        x = self.classifier(x)
        return x


dls = ImageDataLoaders.from_folder(
    untar_data(URLs.MNIST), train="training", valid="testing", num_workers=0
)
learn = Learner(
    dls, SimpleConv(ArcFaceClassifier(3, 10)), metrics=accuracy, loss_func=arcface_loss
)

# %%
learn.fit_one_cycle(5, 5e-3)

# %%


def get_embs(model, dl):
    embs = []
    ys = []
    for bx, by in tqdm(dl):
        with torch.no_grad():
            embs.append(model.get_embs(bx))
            ys.append(by)
    embs = torch.cat(embs)
    embs = embs / embs.norm(p=2, dim=1)[:, None]
    ys = torch.cat(ys)
    return embs, ys


# helper to plot embeddings in 3D
def plot_embs(embs, ys, ax):
    # ax.axis('off')
    for k in range(10):
        e = embs[ys == k].cpu()
        ax.scatter(e[:, 0], e[:, 1], e[:, 2], s=4, alpha=0.2)


embs_arcface, ys_arcface = get_embs(learn.model.eval(), dls.valid)
