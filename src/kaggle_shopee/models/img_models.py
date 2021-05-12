import sys
from typing import Optional

import pandas as pd

sys.path.append("/kaggle/input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0")

import numpy as np
import timm
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from kaggle_shopee.factories.config_factory import (BertPoolingConfig,
                                                    LogitTypeEnum,
                                                    MetricLearningConfig,
                                                    ModelConfig, PoolingConfig)
from kaggle_shopee.factories.data_factory import Data
from kaggle_shopee.factories.metric_learning_factory import \
    MetricLearningFactory
from kaggle_shopee.factories.pooling_factory import PoolingFactory


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ShopeeImgNet(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        pooling_config: PoolingConfig,
    ):
        super(ShopeeImgNet, self).__init__()
        self.model_config = model_config
        channel_size = model_config.channel_size
        self.backbone = timm.create_model(
            model_config.model_arch, pretrained=model_config.pretrained
        )

        if "resnext" in model_config.model_arch or "resnet" in model_config.model_arch:
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif "vit" in model_config.model_arch:
            final_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        self.backbone.global_pool = nn.Identity()
        self.pooling = PoolingFactory.get_pooling(pooling_config)

        self.dropout = nn.Dropout(p=model_config.dropout)
        self.fc = nn.Linear(final_in_features, channel_size)
        self.bn = nn.BatchNorm1d(channel_size)

        self.classifier = nn.Linear(channel_size, out_features)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, labels=None):
        if self.model_config.forward_method == "forward":
            x = self.backbone.forward_features(x)
        else:
            x = self.backbone(x)
        x = self.pooling(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        out = F.normalize(x).float()
        if labels is not None:
            return self.margin(out, labels)
        return out


class ShopeeImgNet2(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        pooling_config: PoolingConfig,
        train_df: pd.DataFrame = pd.DataFrame(),
    ):
        super(ShopeeImgNet2, self).__init__()
        self.model_config = model_config
        self.pooling_config = pooling_config
        self.met_config = met_config
        channel_size = model_config.channel_size
        if "efficientnet-" in model_config.model_arch:
            self.backbone = (
                EfficientNet.from_pretrained(model_config.model_arch)
                if model_config.pretrained
                else EfficientNet.from_name(model_config.model_arch)
            )
        else:
            self.backbone = timm.create_model(
                model_config.model_arch, pretrained=model_config.pretrained
            )

        if "nfnet" in model_config.model_arch or "nf_" in model_config.model_arch:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.global_pool = nn.Identity()
            self.backbone.head.fc = nn.Identity()
        elif (
            "resnext" in model_config.model_arch
            or "resnet" in model_config.model_arch
            or "xception" in model_config.model_arch
            or "resnest" in model_config.model_arch
        ):
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif "efficientnet-" in model_config.model_arch:
            final_in_features = self.backbone._fc.in_features
            self.backbone._dropout = nn.Identity()
            self.backbone._fc = nn.Identity()
            self.backbone._swish = nn.Identity()
        elif (
            "vit" in model_config.model_arch
            or "swin" in model_config.model_arch
            or "mixer" in model_config.model_arch
        ):
            final_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        if (
            "efficientnet-" not in model_config.model_arch
            and "nfnet" not in model_config.model_arch
        ):
            self.backbone.global_pool = nn.Identity()

        if pooling_config.name.lower() == "gem":
            self.pooling = GeM(**pooling_config.params)
        else:
            self.pooling = PoolingFactory.get_pooling(pooling_config)

        self.dropout = nn.Dropout(p=model_config.dropout)
        if (
            "vit" in model_config.model_arch
            or "swin" in model_config.model_arch
            or "mixer" in model_config.model_arch
        ):
            self.bn1 = nn.BatchNorm1d(final_in_features)
        else:
            self.bn1 = nn.BatchNorm2d(final_in_features)
        self.fc = nn.Linear(final_in_features, channel_size)
        self.bn2 = nn.BatchNorm1d(channel_size)

        if met_config.name == "ArcAdaptiveMarginProduct":
            self.margin = MetricLearningFactory.get_metric_learning_product(
                met_config,
                in_features=channel_size,
                out_features=out_features,
                train_df=train_df,
            )
        else:
            self.margin = MetricLearningFactory.get_metric_learning_product(
                met_config,
                in_features=channel_size,
                out_features=out_features,
            )
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x, labels=None):
        if self.model_config.forward_method == "forward":
            x = self.backbone.forward_features(x)
        else:
            x = self.backbone(x)
        x = self.bn1(x)
        x = self.dropout(x)
        if (
            "vit" not in self.model_config.model_arch
            and "swin" not in self.model_config.model_arch
            and "efficientnet-" not in self.model_config.model_arch
            and "mixer" not in self.model_config.model_arch
        ):
            if self.pooling_config.name.lower() == "gem":
                x = self.pooling(x)
                x = x[:, :, 0, 0]
            else:
                x = self.pooling(x).view(x.size(0), -1)
        out = self.fc(x)
        out = self.bn2(out)

        if self.model_config.normalize:
            out = F.normalize(out)
        out = out.float()
        x = x.float()

        if self.met_config.name == "ArcMarginProductSubCenter2":
            return self.margin(out)
        if self.met_config.name == "ArcMarginProduct3":
            return self.margin(out)
        if labels is not None:
            return self.margin(out, labels)
        return out, x


class ShopeeImgNet3(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        pooling_config: PoolingConfig,
    ):
        super(ShopeeImgNet3, self).__init__()
        self.model_config = model_config
        channel_size = model_config.channel_size
        self.backbone = timm.create_model(
            model_config.model_arch, pretrained=model_config.pretrained
        )
        if "resnext" in model_config.model_arch or "resnet" in model_config.model_arch:
            final_in_features = self.backbone.fc.in_features
        elif "vit" in model_config.model_arch:
            final_in_features = self.backbone.head.in_features
        else:
            final_in_features = self.backbone.classifier.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = PoolingFactory.get_pooling(pooling_config)

        self.dropout = nn.Dropout(p=model_config.dropout)
        if "vit" in model_config.model_arch:
            self.bn1 = nn.BatchNorm1d(final_in_features)
        else:
            self.bn1 = nn.BatchNorm2d(final_in_features)
        self.fc = nn.Linear(final_in_features, channel_size)
        self.bn2 = nn.BatchNorm1d(channel_size)

        self.margin = MetricLearningFactory.get_metric_learning_product(
            met_config,
            in_features=channel_size,
            out_features=out_features,
        )
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x, labels=None):
        if self.model_config.forward_method == "forward":
            x = self.backbone.forward_features(x)
        else:
            x = self.backbone(x)
        x = self.bn1(x)
        x = self.dropout(x)
        if "vit" not in self.model_config.model_arch:
            x = self.pooling(x).view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        out = F.normalize(x).float()
        if labels is not None:
            return self.margin(out, labels)
        return out


class ShopeeImgNet4(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        pooling_config: PoolingConfig,
        train_df: pd.DataFrame = pd.DataFrame(),
    ):
        super(ShopeeImgNet4, self).__init__()
        self.model_config = model_config
        self.pooling_config = pooling_config
        self.met_config = met_config
        channel_size = model_config.channel_size
        if "efficientnet-" in model_config.model_arch:
            self.backbone = (
                EfficientNet.from_pretrained(model_config.model_arch)
                if model_config.pretrained
                else EfficientNet.from_name(model_config.model_arch)
            )
        else:
            self.backbone = timm.create_model(
                model_config.model_arch, pretrained=model_config.pretrained
            )

        if (
            "resnext" in model_config.model_arch
            or "resnet" in model_config.model_arch
            or "xception" in model_config.model_arch
            or "resnest" in model_config.model_arch
        ):
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif "efficientnet-" in model_config.model_arch:
            final_in_features = self.backbone._fc.in_features
            self.backbone._dropout = nn.Identity()
            self.backbone._fc = nn.Identity()
            self.backbone._swish = nn.Identity()
        elif "vit" in model_config.model_arch:
            final_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif "nfnet" in model_config.model_arch:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.global_pool = nn.Identity()
            self.backbone.head.fc = nn.Identity()
        else:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        if (
            "efficientnet-" not in model_config.model_arch
            and "nfnet" not in model_config.model_arch
        ):
            self.backbone.global_pool = nn.Identity()

        if pooling_config.name.lower() == "gem":
            self.pooling = GeM(**pooling_config.params)
        else:
            self.pooling = PoolingFactory.get_pooling(pooling_config)

        self.dropout = nn.Dropout(p=model_config.dropout)
        self.bn1 = nn.BatchNorm1d(final_in_features)
        self.fc = nn.Linear(final_in_features, channel_size)
        self.bn2 = nn.BatchNorm1d(channel_size)

        if met_config.name == "ArcAdaptiveMarginProduct":
            self.margin = MetricLearningFactory.get_metric_learning_product(
                met_config,
                in_features=channel_size,
                out_features=out_features,
                train_df=train_df,
            )
        else:
            self.margin = MetricLearningFactory.get_metric_learning_product(
                met_config,
                in_features=channel_size,
                out_features=out_features,
            )
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.pooling(x).view(x.size(0), -1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn2(x)
        out = F.normalize(x).float()
        if labels is not None:
            return self.margin(out, labels)
        return out


class ShopeeImgNet5(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        pooling_config: PoolingConfig,
        train_df: pd.DataFrame = pd.DataFrame(),
    ):
        super(ShopeeImgNet5, self).__init__()
        self.model_config = model_config
        self.pooling_config = pooling_config
        self.met_config = met_config
        channel_size = model_config.channel_size
        self.cnn = timm.create_model(
            model_config.model_arch, pretrained=model_config.pretrained, num_classes=0
        )
        self.dropout = nn.Dropout(p=model_config.dropout)
        self.bn1 = nn.BatchNorm1d(self.cnn.num_features)
        self.fc1 = nn.Linear(self.cnn.num_features, channel_size)
        self.bn2 = nn.BatchNorm1d(channel_size)

        if met_config.name == "ArcAdaptiveMarginProduct":
            self.margin = MetricLearningFactory.get_metric_learning_product(
                met_config,
                in_features=channel_size,
                out_features=out_features,
                train_df=train_df,
            )
        else:
            self.margin = MetricLearningFactory.get_metric_learning_product(
                met_config,
                in_features=channel_size,
                out_features=out_features,
            )
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x, labels=None):
        x = self.cnn(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        if labels is not None:
            return self.margin(x, labels)
        return x
