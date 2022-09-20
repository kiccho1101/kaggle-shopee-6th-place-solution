import gc
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from cuml.neighbors import NearestNeighbors
except ImportError:
    from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from kaggle_shopee.factories.config_factory import Config, ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data
from kaggle_shopee.factories.loss_factory import LossFactory
from kaggle_shopee.factories.metric_learning_factory import MetricLearningFactory
from kaggle_shopee.factories.model_factory import ModelFactory
from kaggle_shopee.factories.optimizer_factory import OptimizerFactory
from kaggle_shopee.factories.scheduler_factory import SchedulerFactory
from kaggle_shopee.models import img_models, txt_models
from kaggle_shopee.utils.file_util import FileUtil
from kaggle_shopee.utils.metric_util import MetricUtil
from kaggle_shopee.utils.mlflow_util import MlflowUtil
from kaggle_shopee.utils.string_util import StringUtil
from kaggle_shopee.utils.test_util import TestUtil
from kaggle_shopee.utils.time_util import TimeUtil
from sklearn.preprocessing import normalize
from torch.nn.parallel.data_parallel import data_parallel
from tqdm.autonotebook import tqdm


class ShopeeImgTextNet2(nn.Module):
    def __init__(
        self,
        config: Config,
        data: Data,
        out_features: int,
        train_df: pd.DataFrame = pd.DataFrame(),
        bert_path: Optional[str] = None,
    ):
        super(ShopeeImgTextNet2, self).__init__()
        self.config = config
        print("img_checkpoint:", config.model_config.img_checkpoint)
        print("txt_checkpoint:", config.model_config.txt_checkpoint)
        img_exp = config.model_config.img_checkpoint.split("_")[0]
        img_config = ConfigFactory.get_config_from_yaml_file(img_exp, config.env, False)
        img_config.model_config.pretrained = False
        img_config.model_config.normalize = False
        txt_exp = config.model_config.txt_checkpoint.split("_")[0]
        txt_config = ConfigFactory.get_config_from_yaml_file(txt_exp, config.env, False)
        txt_config.model_config.pretrained = False
        txt_config.model_config.normalize = False
        img_lit_model: ShopeeLitModel = ShopeeLitModel.load_from_checkpoint(
            os.path.join(
                str(config.dir_config.checkpoint_dir),
                config.model_config.img_checkpoint,
            ),
            data=data,
            config=img_config,
            fold=-1,
            with_mlflow=False,
        )
        txt_lit_model: ShopeeLitModel = ShopeeLitModel.load_from_checkpoint(
            os.path.join(
                str(config.dir_config.checkpoint_dir),
                config.model_config.txt_checkpoint,
            ),
            data=data,
            config=txt_config,
            fold=-1,
            with_mlflow=False,
            bert_path=bert_path,
        )
        self.img_model = img_lit_model.model
        self.txt_model = txt_lit_model.model
        img_out_features = list(self.img_model.children())[-2].num_features
        txt_out_features = list(self.txt_model.children())[-2].num_features
        print("img_out_features:", img_out_features)
        print("txt_out_features:", txt_out_features)
        concat_features = img_out_features + txt_out_features

        self.bn1 = nn.BatchNorm1d(concat_features)
        self.dropout = nn.Dropout(config.model_config.dropout)

        self.fc1 = nn.Linear(concat_features, config.model_config.channel_size)
        self.bn2 = nn.BatchNorm1d(config.model_config.channel_size)
        self._init_params()
        if config.met_config.name == "ArcAdaptiveMarginProduct":
            self.margin = MetricLearningFactory.get_metric_learning_product(
                config.met_config,
                in_features=config.model_config.channel_size,
                out_features=out_features,
                train_df=train_df,
            )
        else:
            self.margin = MetricLearningFactory.get_metric_learning_product(
                config.met_config,
                in_features=config.model_config.channel_size,
                out_features=out_features,
            )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, img, input_ids, attention_mask, labels=None):
        img_out = self.img_model(img)
        txt_out = self.txt_model(input_ids, attention_mask)
        x = torch.cat([img_out, txt_out], 1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)
        if self.config.model_config.normalize:
            out = F.normalize(x).float()
        else:
            out = x.float()
        if labels is not None:
            return self.margin(out, labels)
        return out


class ShopeeImgTextNet3(nn.Module):
    def __init__(
        self,
        config: Config,
        data: Data,
        out_features: int,
        train_df: pd.DataFrame = pd.DataFrame(),
        bert_path: Optional[str] = None,
        is_test: bool = False,
    ):
        super(ShopeeImgTextNet3, self).__init__()
        self.config = config
        print("img_checkpoint:", config.model_config.img_checkpoint)
        print("txt_checkpoint:", config.model_config.txt_checkpoint)

        img_exp = config.model_config.img_checkpoint.split("_")[0]
        img_config = ConfigFactory.get_config_from_yaml_file(img_exp, config.env, False)
        if is_test or config.model_config.with_pretrain:
            img_config.model_config.pretrained = False
        img_config.model_config.normalize = False
        self.img_model = img_models.ShopeeImgNet2(
            out_features,
            img_config.model_config,
            img_config.met_config,
            img_config.pooling_config,
        )

        txt_exp = config.model_config.txt_checkpoint.split("_")[0]
        txt_config = ConfigFactory.get_config_from_yaml_file(txt_exp, config.env, False)
        if is_test or config.model_config.with_pretrain:
            txt_config.model_config.pretrained = False
        txt_config.model_config.normalize = False
        self.txt_model = txt_models.ShopeeTextNet(
            out_features,
            txt_config.model_config,
            txt_config.met_config,
            txt_config.bert_pooling_config,
            bert_path,
        )

        img_out_features = list(self.img_model.children())[-2].num_features
        txt_out_features = list(self.txt_model.children())[-2].num_features
        print("img_out_features:", img_out_features)
        print("txt_out_features:", txt_out_features)
        concat_features = img_out_features + txt_out_features

        self.bn1 = nn.BatchNorm1d(concat_features)
        self.dropout = nn.Dropout(config.model_config.dropout)
        self.fc1 = nn.Linear(concat_features, config.model_config.channel_size)
        self.bn2 = nn.BatchNorm1d(config.model_config.channel_size)
        self._init_params()

        if config.met_config.name == "ArcAdaptiveMarginProduct":
            self.margin = MetricLearningFactory.get_metric_learning_product(
                config.met_config,
                in_features=config.model_config.channel_size,
                out_features=out_features,
                train_df=train_df,
            )
        else:
            self.margin = MetricLearningFactory.get_metric_learning_product(
                config.met_config,
                in_features=config.model_config.channel_size,
                out_features=out_features,
            )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, img, input_ids, attention_mask, labels=None):
        img_out, _ = self.img_model(img)
        txt_out, _ = self.txt_model(input_ids, attention_mask)
        x = torch.cat([img_out, txt_out], 1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)

        if self.config.model_config.normalize:
            out = F.normalize(x).float()
        else:
            out = x.float()

        if labels is not None:
            return self.margin(out, labels)
        return out


class ShopeeImgTextNet4(nn.Module):
    def __init__(
        self,
        config: Config,
        data: Data,
        out_features: int,
        train_df: pd.DataFrame = pd.DataFrame(),
        bert_path: Optional[str] = None,
        is_test: bool = False,
    ):
        super(ShopeeImgTextNet4, self).__init__()
        self.config = config
        print("img_checkpoint:", config.model_config.img_checkpoint)
        print("txt_checkpoint:", config.model_config.txt_checkpoint)

        img_exp = config.model_config.img_checkpoint.split("_")[0]
        img_config = ConfigFactory.get_config_from_yaml_file(img_exp, config.env, False)
        if is_test or config.model_config.with_pretrain:
            img_config.model_config.pretrained = False
        img_config.model_config.normalize = False
        self.img_model = img_models.ShopeeImgNet2(
            out_features,
            img_config.model_config,
            img_config.met_config,
            img_config.pooling_config,
        )
        self.img_margin = self.img_model.margin

        txt_exp = config.model_config.txt_checkpoint.split("_")[0]
        txt_config = ConfigFactory.get_config_from_yaml_file(txt_exp, config.env, False)
        if is_test or config.model_config.with_pretrain:
            txt_config.model_config.pretrained = False
        txt_config.model_config.normalize = False
        self.txt_model = txt_models.ShopeeTextNet(
            out_features,
            txt_config.model_config,
            txt_config.met_config,
            txt_config.bert_pooling_config,
            bert_path,
        )
        self.txt_margin = self.txt_model.margin

        img_out_features = list(self.img_model.children())[-2].num_features
        txt_out_features = list(self.txt_model.children())[-2].num_features
        print("img_out_features:", img_out_features)
        print("txt_out_features:", txt_out_features)
        concat_features = img_out_features + txt_out_features

        self.bn1 = nn.BatchNorm1d(concat_features)
        self.dropout = nn.Dropout(config.model_config.dropout)
        self.fc1 = nn.Linear(concat_features, config.model_config.channel_size)
        self.bn2 = nn.BatchNorm1d(config.model_config.channel_size)
        self._init_params()

        if config.met_config.name == "ArcAdaptiveMarginProduct":
            self.margin = MetricLearningFactory.get_metric_learning_product(
                config.met_config,
                in_features=config.model_config.channel_size,
                out_features=out_features,
                train_df=train_df,
            )
        else:
            self.margin = MetricLearningFactory.get_metric_learning_product(
                config.met_config,
                in_features=config.model_config.channel_size,
                out_features=out_features,
            )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, img, input_ids, attention_mask, labels=None):
        img_out, _ = self.img_model(img)
        txt_out, _ = self.txt_model(input_ids, attention_mask)
        x = torch.cat([img_out, txt_out], 1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)

        if self.config.model_config.normalize:
            out = F.normalize(x).float()
        else:
            out = x.float()

        if labels is not None:
            img_out = self.img_margin(img_out, labels)
            txt_out = self.txt_margin(txt_out, labels)
            concat_out = self.margin(out, labels)
            return out, img_out, txt_out, concat_out
        return out, img_out, txt_out


class ShopeeImgTextNet6(nn.Module):
    def __init__(
        self,
        config: Config,
        data: Data,
        out_features: int,
        train_df: pd.DataFrame = pd.DataFrame(),
        bert_path: Optional[str] = None,
        is_test: bool = False,
    ):
        super(ShopeeImgTextNet6, self).__init__()
        self.config = config
        print("img_checkpoint:", config.model_config.img_checkpoint)
        print("txt_checkpoint:", config.model_config.txt_checkpoint)

        img_exp = config.model_config.img_checkpoint.split("_")[0]
        img_config = ConfigFactory.get_config_from_yaml_file(img_exp, config.env, False)
        if is_test or config.model_config.with_pretrain:
            img_config.model_config.pretrained = False
        img_config.model_config.normalize = False
        self.img_model = img_models.ShopeeImgNet2(
            out_features,
            img_config.model_config,
            img_config.met_config,
            img_config.pooling_config,
        )
        self.img_margin = self.img_model.margin
        if "mixer_b" in img_config.model_config.model_arch:
            self.img_bn = nn.BatchNorm1d(768)
        elif "mixer_l" in img_config.model_config.model_arch:
            self.img_bn = nn.BatchNorm1d(1024)
        else:
            self.img_bn = nn.BatchNorm1d(self.img_model.backbone.num_features)

        txt_exp = config.model_config.txt_checkpoint.split("_")[0]
        txt_config = ConfigFactory.get_config_from_yaml_file(txt_exp, config.env, False)
        if is_test or config.model_config.with_pretrain:
            txt_config.model_config.pretrained = False
        txt_config.model_config.normalize = False
        self.txt_model = txt_models.ShopeeTextNet(
            out_features,
            txt_config.model_config,
            txt_config.met_config,
            txt_config.bert_pooling_config,
            bert_path,
        )
        self.txt_margin = self.txt_model.margin
        self.txt_bn = nn.BatchNorm1d(self.txt_model.bert_model.config.hidden_size)

        if "mixer_b" in img_config.model_config.model_arch:
            img_out_features = 768
        elif "mixer_l" in img_config.model_config.model_arch:
            img_out_features = 1024
        else:
            img_out_features = self.img_model.backbone.num_features
        txt_out_features = self.txt_model.bert_model.config.hidden_size
        print("img_out_features:", img_out_features)
        print("txt_out_features:", txt_out_features)
        concat_features = img_out_features + txt_out_features

        self.bn1 = nn.BatchNorm1d(concat_features)
        self.dropout = nn.Dropout(config.model_config.dropout)
        self.fc1 = nn.Linear(concat_features, config.model_config.channel_size)
        self.bn2 = nn.BatchNorm1d(config.model_config.channel_size)
        self._init_params()

        if config.met_config.name == "ArcAdaptiveMarginProduct":
            self.margin = MetricLearningFactory.get_metric_learning_product(
                config.met_config,
                in_features=config.model_config.channel_size,
                out_features=out_features,
                train_df=train_df,
            )
        else:
            self.margin = MetricLearningFactory.get_metric_learning_product(
                config.met_config,
                in_features=config.model_config.channel_size,
                out_features=out_features,
            )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, img, input_ids, attention_mask, labels=None):
        img_out, img = self.img_model(img)
        txt_out, text = self.txt_model(input_ids, attention_mask)
        img = self.img_bn(img)
        x = torch.cat([img, text], 1)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn2(x)

        if self.config.model_config.normalize:
            out = F.normalize(x).float()
        else:
            out = x.float()

        if labels is not None:
            img_out = self.img_margin(img_out, labels)
            txt_out = self.txt_margin(txt_out, labels)
            concat_out = self.margin(out, labels)
            return out, img_out, txt_out, concat_out
        return out, img_out, txt_out


def load_model(checkpoint: str, env: EnvEnum, data: Data, fold: int):
    exp = checkpoint.split("_")[0]
    _checkpoint = FileUtil.get_best_cv_checkpoint(env, exp, fold)
    print("load model:", _checkpoint)
    config = ConfigFactory.get_config_from_yaml_file(exp, env, False)
    config.model_config.pretrained = False
    config.model_config.normalize = False
    lit_model: ShopeeLitModel = ShopeeLitModel.load_from_checkpoint(
        os.path.join(str(config.dir_config.checkpoint_dir), _checkpoint),
        data=data,
        config=config,
        fold=fold,
        with_mlflow=False,
    )
    return lit_model.model


class ShopeeLitModel(pl.LightningModule):
    def __init__(
        self,
        data: Data,
        config: Config,
        fold: int = 0,
        with_mlflow: bool = True,
        bert_path: Optional[str] = None,
        is_test: bool = False,
    ):
        super(ShopeeLitModel, self).__init__()
        self.config = config
        self.fold = fold
        self.with_mlflow = with_mlflow
        self.data = data
        self.train_df = data.train.copy()
        self.valid_df = self.train_df[
            self.train_df[config.cv_config.fold_col] == fold
        ].copy()
        self.model_type = StringUtil.get_model_type(config.model_config.model_name)
        self.n_classes = self.train_df[config.train_config.target].nunique()
        if config.model_config.model_name == "ShopeeImgTextNet2":
            img_exp = config.model_config.img_checkpoint.split("_")[0]
            txt_exp = config.model_config.txt_checkpoint.split("_")[0]
            config.model_config.img_checkpoint = FileUtil.get_best_cv_checkpoint(
                config.env, img_exp, fold
            )
            config.model_config.txt_checkpoint = FileUtil.get_best_cv_checkpoint(
                config.env, txt_exp, fold
            )
            self.model = ShopeeImgTextNet2(
                config,
                data,
                self.n_classes,
                train_df=self.train_df,
                bert_path=bert_path,
            )
        elif config.model_config.model_name == "ShopeeImgTextNet3":
            self.model = ShopeeImgTextNet3(
                config,
                data,
                self.n_classes,
                train_df=self.train_df,
                bert_path=bert_path,
                is_test=is_test,
            )
            if not is_test and config.model_config.with_pretrain:
                self.model.img_model = load_model(
                    config.model_config.img_checkpoint, config.env, data, fold
                )
                self.model.txt_model = load_model(
                    config.model_config.txt_checkpoint, config.env, data, fold
                )
        elif config.model_config.model_name == "ShopeeImgTextNet4":
            self.model = ShopeeImgTextNet4(
                config,
                data,
                self.n_classes,
                train_df=self.train_df,
                bert_path=bert_path,
                is_test=is_test,
            )
            if not is_test and config.model_config.with_pretrain:
                self.model.img_model = load_model(
                    config.model_config.img_checkpoint, config.env, data, fold
                )
                self.model.txt_model = load_model(
                    config.model_config.txt_checkpoint, config.env, data, fold
                )
        elif config.model_config.model_name == "ShopeeImgTextNet6":
            self.model = ShopeeImgTextNet6(
                config,
                data,
                self.n_classes,
                train_df=self.train_df,
                bert_path=bert_path,
                is_test=is_test,
            )
            if not is_test and config.model_config.with_pretrain:
                self.model.img_model = load_model(
                    config.model_config.img_checkpoint, config.env, data, fold
                )
                self.model.txt_model = load_model(
                    config.model_config.txt_checkpoint, config.env, data, fold
                )
        else:
            self.model = ModelFactory.get_model(
                out_features=self.n_classes,
                model_config=config.model_config,
                met_config=config.met_config,
                pooling_config=config.pooling_config,
                bert_pooling_config=config.bert_pooling_config,
                bert_path=bert_path,
                train_df=self.train_df,
            )
        if config.loss_config.name == "CrossEntropyLossWithLabelSmoothing":
            self.criterion = LossFactory.get_loss(
                config.loss_config,
                n_dim=self.n_classes,
            )
        elif config.loss_config.name == "ArcFaceLossAdaptiveMargin":
            self.criterion = LossFactory.get_loss(
                config.loss_config,
                train_df=self.train_df,
                out_dim=self.n_classes,
            )
        else:
            self.criterion = LossFactory.get_loss(config.loss_config)
        self.triplet_criterion = nn.TripletMarginLoss(margin=0.1, p=2.0, eps=1e-6)
        self.y_pred = []
        self.y_true = []
        self.features = []
        self.img_features = []
        self.txt_features = []
        if self.with_mlflow:
            self.best_score = MlflowUtil.get_best_score(config, fold)
        else:
            self.best_score = -1e10

    def forward(self, x, labels=None):
        out = self.model(x, labels)
        return out

    def on_train_epoch_start(self) -> None:
        self.y_pred = []
        self.y_true = []
        self.features = []
        self.img_features = []
        self.txt_features = []

    def on_validation_epoch_start(self) -> None:
        self.y_pred = []
        self.y_true = []
        self.features = []
        self.img_features = []
        self.txt_features = []

    def step(self, batch: Dict[str, torch.Tensor], mode: str = "train"):
        if self.config.model_config.model_name in [
            "ShopeeImgTextNet4",
            "ShopeeImgTextNet6",
            "ShopeeVitImgTextNet",
        ]:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            img = batch["img"]
            label = batch["label"]
            with torch.cuda.amp.autocast():
                if mode == "train":
                    logit, img_logit, txt_logit, concat_logit = self.model(
                        img, input_ids, attention_mask, label
                    )
                    concat_loss = self.criterion(concat_logit, label)
                    img_loss = self.criterion(img_logit, label)
                    txt_loss = self.criterion(txt_logit, label)

                    w_concat, w_img, w_txt = (
                        self.config.model_config.w_concat,
                        self.config.model_config.w_img,
                        self.config.model_config.w_txt,
                    )
                    loss = w_concat * concat_loss + w_img * img_loss + w_txt * txt_loss

                    if self.config.dataset_config.is_with_triplets():
                        w_triplet = self.config.model_config.w_triplet
                        p_img = batch["positive"]["img"]
                        p_input_ids = batch["positive"]["input_ids"]
                        p_attention_mask = batch["positive"]["attention_mask"]
                        n_img = batch["negative"]["img"]
                        n_input_ids = batch["negative"]["input_ids"]
                        n_attention_mask = batch["negative"]["attention_mask"]
                        p_logit, _, _ = self.model(p_img, p_input_ids, p_attention_mask)
                        n_logit, _, _ = self.model(n_img, n_input_ids, n_attention_mask)
                        triplet_loss = self.triplet_criterion(logit, p_logit, n_logit)
                        loss += w_triplet * triplet_loss
                else:
                    logit, img_logit, txt_logit = self.model(
                        img, input_ids, attention_mask
                    )
                    self.img_features += [img_logit.detach().cpu()]
                    self.txt_features += [txt_logit.detach().cpu()]
                    loss = -1
        elif self.config.model_config.model_name == "ShopeeImgTextNet5":
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            img = batch["img"]
            label = batch["label"]
            with torch.cuda.amp.autocast():
                if mode == "train":
                    logit, global_out, local_out, bert_out, concat_out = self.model(
                        img, input_ids, attention_mask, label
                    )
                    global_loss = self.criterion(global_out, label)
                    local_loss = self.criterion(local_out, label)
                    bert_loss = self.criterion(bert_out, label)
                    concat_loss = self.criterion(concat_out, label)
                    w_concat, w_img, w_img_local, w_txt = (
                        self.config.model_config.w_concat,
                        self.config.model_config.w_img,
                        self.config.model_config.w_img_local,
                        self.config.model_config.w_txt,
                    )
                    loss = (
                        w_concat * concat_loss
                        + w_img * global_loss
                        + w_img_local * local_loss
                        + w_txt * bert_loss
                    )
                else:
                    logit = self.model(img, input_ids, attention_mask)
                    loss = -1
        elif self.model_type == "image+text":
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            img = batch["img"]
            label = batch["label"]
            with torch.cuda.amp.autocast():
                if mode == "train":
                    logit = self.model(img, input_ids, attention_mask, label)
                    loss = self.criterion(logit, label)
                else:
                    logit = self.model(img, input_ids, attention_mask)
                    loss = -1
        elif self.model_type == "image":
            img = batch["img"]
            label = batch["label"]
            with torch.cuda.amp.autocast():
                if mode == "train":
                    logit = self.model(img, label)
                    loss = self.criterion(logit, label)
                else:
                    logit, _ = self.model(img)
                    loss = -1
                logit = F.normalize(logit)
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            label = batch["label"]
            with torch.cuda.amp.autocast():
                if mode == "train":
                    logit = self.model(input_ids, attention_mask, label)
                    loss = self.criterion(logit, label)
                    if self.config.dataset_config.is_with_triplets():
                        w_triplet = self.config.model_config.w_triplet
                        p_input_ids = batch["positive"]["input_ids"]
                        p_attention_mask = batch["positive"]["attention_mask"]
                        n_input_ids = batch["negative"]["input_ids"]
                        n_attention_mask = batch["negative"]["attention_mask"]
                        p_logit, _, _ = self.model(p_input_ids, p_attention_mask)
                        n_logit, _, _ = self.model(n_input_ids, n_attention_mask)
                        triplet_loss = self.triplet_criterion(logit, p_logit, n_logit)
                        loss += w_triplet * triplet_loss
                else:
                    logit, _ = self.model(input_ids, attention_mask)
                    loss = -1
        self.features += [logit.detach().cpu()]
        if mode != "test":
            self.y_pred += [torch.argmax(logit, 1).detach().cpu()]
            self.y_true += [label.detach().cpu()]
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self.step(batch, "train")
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self.step(batch, "valid")
        return {"valid_loss": loss}

    def _find_matches(
        self,
        posting_ids: np.ndarray,
        threshold: float,
        features: np.ndarray,
        n_batches: int,
    ) -> List[List[str]]:
        TestUtil.assert_any(len(posting_ids), len(features))
        sim_threshold = threshold / 100

        y_pred: List[List[str]] = []
        n_rows = features.shape[0]
        bs = n_rows // n_batches
        batches = []
        for i in range(n_batches):
            left = bs * i
            right = bs * (i + 1)
            if i == n_batches - 1:
                right = n_rows
            batches.append(features[left:right, :])
        for batch in batches:
            dot_product = batch @ features.T
            selection = dot_product > sim_threshold
            for j in range(len(selection)):
                IDX = selection[j]
                if np.sum(IDX) < self.config.inference_config.min_indices:
                    IDX = np.argsort(dot_product[j])[
                        -self.config.inference_config.min_indices :
                    ]
                y_pred.append(posting_ids[IDX].tolist())

        TestUtil.assert_any(len(posting_ids), len(y_pred))
        return y_pred

    def _find_threshold(
        self, features, thresholds: np.ndarray = np.arange(40, 100, 5)
    ) -> Tuple[float, float, List[List[str]]]:
        features = F.normalize(torch.from_numpy(features)).numpy()
        best_score = 0
        best_threshold = -1
        best_y_pred: List[List[str]] = []
        print()
        for threshold in thresholds:
            y_pred = self._find_matches(
                self.valid_df["posting_id"].values, threshold, features, n_batches=10
            )
            if len(y_pred[0]) == len(self.valid_df):
                print(
                    f"----------- can't calculate f1 score. threshold: {threshold} ------------"
                )
            else:
                scores = MetricUtil.f1_scores(self.valid_df["target"].tolist(), y_pred)
                precisions, recalls = MetricUtil.precision_recall(
                    self.valid_df["target"].tolist(), y_pred
                )
                self.valid_df["score"] = scores
                self.valid_df["precision"] = precisions
                self.valid_df["recall"] = recalls
                selected_score = self.valid_df["score"].mean()
                _p_mean = self.valid_df["precision"].mean()
                _r_mean = self.valid_df["recall"].mean()
                print(
                    f"----------- valid f1: {selected_score} precision: {_p_mean} recall: {_r_mean} threshold: {threshold} ------------"
                )
                if selected_score > best_score:
                    best_score = selected_score
                    best_threshold = threshold
                    best_y_pred = y_pred
        return best_score, best_threshold, best_y_pred

    def _find_distance_threshold(
        self,
        features,
        posting_ids: np.ndarray,
        thresholds: List[float],
    ) -> Tuple[float, float, List[List[str]]]:
        features = F.normalize(torch.from_numpy(features)).numpy()
        with TimeUtil.timer("nearest neighbor search"):
            model = NearestNeighbors(n_neighbors=len(self.valid_df), n_jobs=32)
            model.fit(features)
            distances, indices = model.kneighbors(features)
            FileUtil.save_npy(
                distances,
                self.config.dir_config.output_dir
                / f"distances_{self.fold}_{self.current_epoch:02d}.npy",
            )
            FileUtil.save_npy(
                indices,
                self.config.dir_config.output_dir
                / f"indices_{self.fold}_{self.current_epoch:02d}.npy",
            )
        best_score = 0
        best_threshold = -1
        best_y_pred: List[List[str]] = []
        for threshold in thresholds:
            y_pred = []
            for i in range(len(distances)):
                IDX = np.where(distances[i] < threshold)[0]
                if len(IDX) < self.config.inference_config.min_indices:
                    IDX = list(range(self.config.inference_config.min_indices))
                idxs = indices[i, IDX]
                y_pred.append(posting_ids[idxs])
            scores = MetricUtil.f1_scores(self.valid_df["target"].tolist(), y_pred)
            precisions, recalls = MetricUtil.precision_recall(
                self.valid_df["target"].tolist(), y_pred
            )
            self.valid_df["score"] = scores
            self.valid_df["precision"] = precisions
            self.valid_df["recall"] = recalls
            selected_score = self.valid_df["score"].mean()
            _p_mean = self.valid_df["precision"].mean()
            _r_mean = self.valid_df["recall"].mean()
            print(
                f"----------- valid f1: {selected_score} precision: {_p_mean} recall: {_r_mean} threshold: {threshold} ------------"
            )
            if selected_score > best_score:
                best_score = selected_score
                best_threshold = threshold
                best_y_pred = y_pred
        return best_score, best_threshold, best_y_pred

    def _validation(self):
        features = torch.cat(self.features).cpu().numpy()
        FileUtil.save_npy(
            features,
            self.config.dir_config.output_dir
            / f"features_{self.fold}_{self.current_epoch:02d}.npy",
        )
        if self.config.model_config.model_name in [
            "ShopeeImgTextNet4",
            "ShopeeImgTextNet6",
            "ShopeeVitImgTextNet",
        ]:
            img_features = torch.cat(self.img_features).cpu().numpy()
            txt_features = torch.cat(self.txt_features).cpu().numpy()
            FileUtil.save_npy(
                img_features,
                self.config.dir_config.output_dir
                / f"img_features_{self.fold}_{self.current_epoch:02d}.npy",
            )
            FileUtil.save_npy(
                txt_features,
                self.config.dir_config.output_dir
                / f"txt_features_{self.fold}_{self.current_epoch:02d}.npy",
            )
        if self.config.inference_config.threshold_search_method == "nearest-neighbors":
            score, threshold, y_pred = self._find_distance_threshold(
                features,
                self.valid_df["posting_id"].values,
                thresholds=(np.arange(10, 110, 5) / 100).tolist(),
            )
        else:
            score, threshold, y_pred = self._find_threshold(
                features,
                thresholds=np.arange(10, 90, 5)
                if self.config.train_config.target == "category"
                else np.arange(30, 100, 5),
            )
        self.valid_df["y_pred"] = y_pred
        print()
        print(
            f"----------- {self.config.exp} Fold {self.fold} Epoch {self.current_epoch} ------------"
        )
        print(f"----------- valid best f1 {score} threshold: {threshold} ------------")
        if self.best_score < score:
            self.best_score = score
        self.valid_df.to_pickle(
            self.config.dir_config.output_dir / f"valid_df_{self.fold}_{self.current_epoch:02d}.pickle",
        )
        print()
        return score, threshold

    def validation_epoch_end(self, outputs):
        score, threshold = self._validation()
        self.log("valid_f1", score)
        self.log("threshold", threshold)
        return {"valid_f1": score, "threshold": threshold}

    def configure_optimizers(self):
        print(
            "model_type:",
            StringUtil.get_model_type(self.config.model_config.model_name),
        )
        if self.config.model_config.model_name == "ShopeeImgTextNet":
            optimizer = torch.optim.Adam(
                params=[
                    {
                        "params": self.model.backbone.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                    {
                        "params": self.model.bert_model.parameters(),
                        "lr": self.config.optimizer_config.params["bert_lr"],
                    },
                    {
                        "params": self.model.bn1.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                    {
                        "params": self.model.fc1.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                    {
                        "params": self.model.bn2.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                    {
                        "params": self.model.fc2.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                    {
                        "params": self.model.bn3.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                    {
                        "params": self.model.margin.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                ]
            )
        elif self.config.model_config.model_name == "ShopeeKurupicalExp044Net":
            optimizer = torch.optim.Adam(
                params=[
                    {
                        "params": self.model.bert.parameters(),
                        "lr": self.config.optimizer_config.params["bert_lr"],
                    },
                    {
                        "params": self.model.cnn.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                    {
                        "params": self.model.cnn_bn.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                    {
                        "params": self.model.fc.parameters(),
                        "lr": self.config.optimizer_config.params["fc_lr"],
                    },
                    {
                        "params": self.model.final.parameters(),
                        "lr": self.config.optimizer_config.params["fc_lr"],
                    },
                ]
            )
        elif self.config.model_config.model_name == "ShopeeImgTextNet2":
            params = [
                {
                    "params": self.model.img_model.parameters(),
                    "lr": self.config.optimizer_config.params["cnn_lr"],
                },
                {
                    "params": self.model.txt_model.parameters(),
                    "lr": self.config.optimizer_config.params["bert_lr"],
                },
                {
                    "params": self.model.bn1.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.fc1.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.bn2.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.margin.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
            ]
            if self.config.model_config.n_conv > 0:
                params.append(
                    {
                        "params": self.model.conv.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                )
            optimizer = torch.optim.Adam(params=params)
        elif self.config.model_config.model_name == "ShopeeImgTextNet3":
            params = [
                {
                    "params": self.model.img_model.parameters(),
                    "lr": self.config.optimizer_config.params["cnn_lr"],
                },
                {
                    "params": self.model.txt_model.parameters(),
                    "lr": self.config.optimizer_config.params["bert_lr"],
                },
                {
                    "params": self.model.bn1.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.fc1.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.bn2.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.margin.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
            ]
            if self.config.model_config.n_conv > 0:
                params.append(
                    {
                        "params": self.model.conv.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                )
            optimizer = torch.optim.Adam(params=params)
        elif self.config.model_config.model_name == "ShopeeImgTextNet4":
            params = [
                {
                    "params": self.model.img_model.parameters(),
                    "lr": self.config.optimizer_config.params["cnn_lr"],
                },
                {
                    "params": self.model.txt_model.parameters(),
                    "lr": self.config.optimizer_config.params["bert_lr"],
                },
                {
                    "params": self.model.bn1.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.fc1.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.bn2.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.margin.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
            ]
            if self.config.model_config.n_conv > 0:
                params.append(
                    {
                        "params": self.model.conv.parameters(),
                        "lr": self.config.optimizer_config.params["lr"],
                    },
                )
            optimizer = torch.optim.Adam(params=params)
        elif self.config.model_config.model_name == "ShopeeVitImgTextNet":
            cnn_lr = self.config.optimizer_config.params["cnn_lr"]
            bert_lr = self.config.optimizer_config.params["bert_lr"]
            fc_lr = self.config.optimizer_config.params["fc_lr"]
            optimizer = torch.optim.Adam(
                params=[
                    {"params": self.model.bert.parameters(), "lr": bert_lr},
                    {"params": self.model.vit.parameters(), "lr": cnn_lr},
                    {"params": self.model.fc1.parameters(), "lr": fc_lr},
                    {"params": self.model.fc2.parameters(), "lr": fc_lr},
                    {"params": self.model.fc3.parameters(), "lr": fc_lr},
                    {"params": self.model.bn1.parameters(), "lr": fc_lr},
                    {"params": self.model.bn2.parameters(), "lr": fc_lr},
                    {"params": self.model.bn3.parameters(), "lr": fc_lr},
                    {"params": self.model.margin.parameters(), "lr": fc_lr},
                ]
            )
        elif self.config.model_config.model_name == "ShopeeImgTextNet5":
            params = [
                {
                    "params": self.model.cnn.parameters(),
                    "lr": self.config.optimizer_config.params["cnn_lr"],
                },
                {
                    "params": self.model.bert.parameters(),
                    "lr": self.config.optimizer_config.params["bert_lr"],
                },
                {
                    "params": self.model.fc_global.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.bn_global.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.fc_local.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.bn_local.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.fc_bert.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.bn_bert.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.fc_concat.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.bn_concat.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
            ]
            optimizer = torch.optim.Adam(params=params)
        elif self.config.model_config.model_name == "ShopeeImgTextNet6":
            params = [
                {
                    "params": self.model.img_model.parameters(),
                    "lr": self.config.optimizer_config.params["cnn_lr"],
                },
                {
                    "params": self.model.txt_model.parameters(),
                    "lr": self.config.optimizer_config.params["bert_lr"],
                },
                {
                    "params": self.model.bn1.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.fc1.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.bn2.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.img_bn.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.txt_bn.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
                {
                    "params": self.model.margin.parameters(),
                    "lr": self.config.optimizer_config.params["lr"],
                },
            ]
            amsgrad = False
            if "amsgrad" in self.config.optimizer_config.params:
                amsgrad = self.config.optimizer_config.params["amsgrad"]
            if self.config.optimizer_config.name == "AdamW":
                optimizer = torch.optim.AdamW(params=params, amsgrad=amsgrad)
            else:
                optimizer = torch.optim.Adam(params=params, amsgrad=amsgrad)
        else:
            optimizer = OptimizerFactory.get_optimizer(
                self.config.optimizer_config, self.parameters()
            )
        scheduler = SchedulerFactory.get_scheduler(
            self.config.scheduler_config, optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_f1",
        }
