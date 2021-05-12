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
                                                    MetricLearningConfig,
                                                    ModelConfig, PoolingConfig)
from kaggle_shopee.factories.metric_learning_factory import \
    MetricLearningFactory
from kaggle_shopee.factories.pooling_factory import PoolingFactory
from transformers import AutoModel


class ShopeeImgTextNet(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        pooling_config: PoolingConfig,
        bert_pooling_config: BertPoolingConfig,
        bert_path: Optional[str] = None,
    ):
        super(ShopeeImgTextNet, self).__init__()
        self.model_config = model_config
        self.bert_pooling_config = bert_pooling_config
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
        self.dropout1 = nn.Dropout(p=model_config.dropout)
        self.bn1 = nn.BatchNorm2d(final_in_features)

        self.bert_model = AutoModel.from_pretrained(
            model_config.bert_model_arch if bert_path is None else bert_path
        )
        self.fc1 = nn.Linear(
            model_config.bert_hidden_size, model_config.bert_hidden_size
        )
        self.bn2 = nn.BatchNorm1d(model_config.bert_hidden_size)
        self.dropout2 = nn.Dropout(p=model_config.dropout)

        self.fc2 = nn.Linear(final_in_features + model_config.bert_hidden_size, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self._init_params()

        self.margin = MetricLearningFactory.get_metric_learning_product(
            met_config,
            in_features=512,
            out_features=out_features,
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)

    def forward(self, img, input_ids, attention_mask, labels=None):
        x = self.backbone(img)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.pooling(x).view(x.size(0), -1)

        output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if self.bert_pooling_config.name == "hs-mean":
            hs = output.hidden_states
            hs_idxs = self.bert_pooling_config.params["hs_idxs"]
            seq_output = torch.stack([hs[idx] for idx in hs_idxs]).mean(dim=0)
            avg_output = torch.sum(
                seq_output * attention_mask.unsqueeze(-1), dim=1, keepdim=False
            )
            avg_output = avg_output / torch.sum(attention_mask, dim=-1, keepdim=True)
            y = avg_output
        else:
            last_hidden_state = output.last_hidden_state
            y = last_hidden_state[:, 0, :]
        y = self.fc1(y)
        y = self.bn2(y)

        z = torch.cat([x, y], 1)
        z = self.fc2(z)
        z = self.bn3(z)

        if self.model_config.normalize:
            out = F.normalize(z).float()
        else:
            out = z.float()
        if labels is not None:
            return self.margin(out, labels)
        return out


class BertModule(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        bert_path: Optional[str] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.bert = AutoModel.from_pretrained(
            model_config.bert_model_arch if bert_path is None else bert_path
        )
        self.dropout_nlp = nn.Dropout(model_config.dropout_nlp)
        self.hidden_size = self.bert.config.hidden_size
        self.bert_bn = nn.BatchNorm1d(self.hidden_size)
        self.dropout_stack = nn.Dropout(model_config.dropout_bert_stack)

    def forward(self, input_ids, attention_mask):
        text = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )[2]

        text = torch.stack([self.dropout_stack(x) for x in text[-4:]]).mean(dim=0)
        text = torch.sum(text * attention_mask.unsqueeze(-1), dim=1, keepdim=False)
        text = text / torch.sum(attention_mask, dim=-1, keepdim=True)
        text = self.bert_bn(text)
        text = self.dropout_nlp(text)
        return text


class ShopeeKurupicalExp044Net(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        bert_path: Optional[str] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.bert = BertModule(model_config, bert_path)
        self.cnn = timm.create_model(
            model_config.model_arch, pretrained=model_config.pretrained, num_classes=0
        )
        self.cnn_bn = nn.BatchNorm1d(self.cnn.num_features)

        n_feat_concat = self.cnn.num_features + self.bert.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(n_feat_concat, model_config.channel_size),
            nn.BatchNorm1d(model_config.channel_size),
        )
        self.dropout_cnn = nn.Dropout(model_config.dropout)
        self.final = MetricLearningFactory.get_metric_learning_product(
            met_config,
            in_features=model_config.channel_size,
            out_features=out_features,
        )

    def forward(self, X_image, input_ids, attention_mask, label=None):
        x = self.cnn(X_image)
        x = self.cnn_bn(x)
        x = self.dropout_cnn(x)

        text = self.bert(input_ids, attention_mask)
        x = torch.cat([x, text], dim=1)
        ret = self.fc(x)

        if label is not None:
            x = self.final(ret, label)
            return x, ret
        else:
            return ret


class ShopeeVitImgTextNet(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        bert_pooling_config: BertPoolingConfig,
        bert_path: Optional[str] = None,
    ):
        super(ShopeeVitImgTextNet, self).__init__()
        self.model_config = model_config
        self.bert_pooling_config = bert_pooling_config
        self.vit = timm.create_model(
            model_config.model_arch, pretrained=model_config.pretrained, num_classes=0
        )

        vit_num_features = self.vit.num_features
        self.dropout1 = nn.Dropout(p=model_config.dropout)
        self.fc1 = nn.Linear(vit_num_features, model_config.channel_size)
        self.bn1 = nn.BatchNorm1d(vit_num_features)

        self.bert = AutoModel.from_pretrained(
            model_config.bert_model_arch if bert_path is None else bert_path
        )
        self.fc2 = nn.Linear(self.bert.config.hidden_size, model_config.channel_size)
        self.bn2 = nn.BatchNorm1d(self.bert.config.hidden_size)
        self.dropout2 = nn.Dropout(p=model_config.dropout_nlp)

        self.fc3 = nn.Linear(model_config.channel_size * 2, model_config.channel_size)
        self.bn3 = nn.BatchNorm1d(model_config.channel_size)

        self.fc_cnn_out = nn.Sequential(
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.channel_size, model_config.channel_size),
            nn.BatchNorm1d(model_config.channel_size),
        )
        self.fc_text_out = nn.Sequential(
            nn.Dropout(model_config.dropout_nlp),
            nn.Linear(model_config.channel_size, model_config.channel_size),
            nn.BatchNorm1d(model_config.channel_size),
        )

        self.margin = MetricLearningFactory.get_metric_learning_product(
            met_config,
            in_features=model_config.channel_size,
            out_features=out_features,
        )

    def forward(self, img, input_ids, attention_mask, labels=None):
        x = self.vit(img)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc1(x)

        hs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states
        hs_idxs = self.bert_pooling_config.params["hs_idxs"]
        seq_output = torch.stack([hs[idx] for idx in hs_idxs]).mean(dim=0)
        avg_output = torch.sum(
            seq_output * attention_mask.unsqueeze(-1), dim=1, keepdim=False
        )
        avg_output = avg_output / torch.sum(attention_mask, dim=-1, keepdim=True)
        y = avg_output
        y = self.bn2(y)
        y = self.fc2(y)

        z = torch.cat([x, y], 1)
        z = self.fc3(z)

        x = self.fc_cnn_out(x)
        y = self.fc_text_out(y)

        x = x.float()
        y = y.float()
        z = z.float()

        if labels is not None:
            x_out = self.margin(x, labels)
            y_out = self.margin(y, labels)
            z_out = self.margin(z, labels)
            return z, x_out, y_out, z_out
        return z, x, y


class ShopeeImgTextNet5(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        pooling_config: PoolingConfig,
        bert_pooling_config: BertPoolingConfig,
        bert_path: Optional[str] = None,
    ):
        super(ShopeeImgTextNet5, self).__init__()
        self.model_config = model_config

        self.cnn = timm.create_model(
            model_config.model_arch, pretrained=True, num_classes=0
        )
        self.cnn.global_pool = nn.Identity()

        n_features_global = self.cnn.num_features
        self.dropout_global = nn.Dropout(model_config.dropout_global)
        self.fc_global = nn.Linear(n_features_global, model_config.channel_size)
        self.bn_global = nn.BatchNorm1d(model_config.channel_size)

        n_features_local = self.cnn.num_features
        self.dropout_local = nn.Dropout(model_config.dropout_local)
        self.fc_local = nn.Linear(n_features_local, model_config.channel_size)
        self.bn_local = nn.BatchNorm1d(model_config.channel_size)

        self.bert = AutoModel.from_pretrained(
            model_config.bert_model_arch if bert_path is None else bert_path,
        )
        self.dropout_nlp = nn.Dropout(model_config.dropout_nlp)
        n_features_bert = self.bert.config.hidden_size
        self.fc_bert = nn.Linear(n_features_bert, model_config.channel_size)
        self.bn_bert = nn.BatchNorm1d(model_config.channel_size)

        n_features_concat = model_config.channel_size * 3
        self.dropout_concat = nn.Dropout(model_config.dropout)
        self.fc_concat = nn.Linear(n_features_concat, model_config.channel_size)
        self.bn_concat = nn.BatchNorm1d(model_config.channel_size)

        self._init_params()

        self.global_margin = MetricLearningFactory.get_metric_learning_product(
            met_config,
            in_features=model_config.channel_size,
            out_features=out_features,
        )
        self.local_margin = MetricLearningFactory.get_metric_learning_product(
            met_config,
            in_features=model_config.channel_size,
            out_features=out_features,
        )
        self.bert_margin = MetricLearningFactory.get_metric_learning_product(
            met_config,
            in_features=model_config.channel_size,
            out_features=out_features,
        )
        self.concat_margin = MetricLearningFactory.get_metric_learning_product(
            met_config,
            in_features=model_config.channel_size,
            out_features=out_features,
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc_local.weight)
        nn.init.constant_(self.fc_local.bias, 0)
        nn.init.constant_(self.bn_local.weight, 1)
        nn.init.constant_(self.bn_local.bias, 0)

        nn.init.xavier_normal_(self.fc_global.weight)
        nn.init.constant_(self.fc_global.bias, 0)
        nn.init.constant_(self.bn_global.weight, 1)
        nn.init.constant_(self.bn_global.bias, 0)

        nn.init.xavier_normal_(self.fc_bert.weight)
        nn.init.constant_(self.fc_bert.bias, 0)
        nn.init.constant_(self.bn_bert.weight, 1)
        nn.init.constant_(self.bn_bert.bias, 0)

        nn.init.xavier_normal_(self.fc_concat.weight)
        nn.init.constant_(self.fc_concat.bias, 0)
        nn.init.constant_(self.bn_concat.weight, 1)
        nn.init.constant_(self.bn_concat.bias, 0)

    def forward(self, img, input_ids, attention_mask, labels=None):
        feat = self.cnn(img)

        # global feat
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = self.dropout_global(global_feat)
        global_feat = self.fc_global(global_feat)
        global_feat = self.bn_global(global_feat)
        global_feat = global_feat.float()

        # local feat
        local_feat = torch.mean(feat, -1, keepdim=False)

        # local_feat = torch.norm(local_feat, 2, -1, keepdim=False)
        local_feat = torch.mean(local_feat, -1, keepdim=False)

        local_feat = self.dropout_local(local_feat)
        local_feat = self.fc_local(local_feat)
        local_feat = self.bn_local(local_feat)
        local_feat = local_feat.float()

        # bert feat
        hs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states
        bert_feat = torch.stack([hs[idx] for idx in [-1, -2, -3, -4]]).mean(dim=0)
        bert_feat = torch.sum(
            bert_feat * attention_mask.unsqueeze(-1), dim=1, keepdim=False
        )
        bert_feat = bert_feat / torch.sum(attention_mask, dim=-1, keepdim=True)
        bert_feat = self.dropout_nlp(bert_feat)
        bert_feat = self.fc_bert(bert_feat)
        bert_feat = self.bn_bert(bert_feat)
        bert_feat = bert_feat.float()

        # concat feat
        concat_feat = torch.cat([global_feat, local_feat, bert_feat], 1)
        concat_feat = self.dropout_concat(concat_feat)
        concat_feat = self.fc_concat(concat_feat)
        concat_feat = self.bn_concat(concat_feat)
        concat_feat = concat_feat.float()

        if labels is not None:
            global_out = self.global_margin(global_feat, labels)
            local_out = self.local_margin(local_feat, labels)
            bert_out = self.bert_margin(bert_feat, labels)
            concat_out = self.concat_margin(concat_feat, labels)
            return concat_feat, global_out, local_out, bert_out, concat_out

        return concat_feat
