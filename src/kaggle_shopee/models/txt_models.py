from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from kaggle_shopee.factories.config_factory import (
    BertPoolingConfig,
    LogitTypeEnum,
    MetricLearningConfig,
    ModelConfig,
    PoolingConfig,
)
from kaggle_shopee.factories.data_factory import Data
from kaggle_shopee.factories.metric_learning_factory import MetricLearningFactory
from kaggle_shopee.factories.pooling_factory import BertPoolingFactory, PoolingFactory
from torch.nn.modules import pooling
from transformers import AutoConfig, AutoModel


class ShopeeTextNet(nn.Module):
    def __init__(
        self,
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        bert_pooling_config: BertPoolingConfig,
        bert_path: Optional[str] = None,
        train_df: pd.DataFrame = pd.DataFrame(),
    ):
        super().__init__()
        self.model_config = model_config
        self.bert_pooling_config = bert_pooling_config
        config = AutoConfig.from_pretrained(
            model_config.bert_model_arch if bert_path is None else bert_path,
            output_hidden_states=True,
        )
        self.bert_model = AutoModel.from_pretrained(
            model_config.bert_model_arch if bert_path is None else bert_path,
            cache_dir=None,
            config=config,
        )

        if bert_pooling_config.name == "concat-conv":
            self.conv = nn.Conv1d(
                self.bert_model.config.hidden_size
                * len(bert_pooling_config.params["hs_idxs"]),
                self.bert_model.config.hidden_size,
                3,
                padding=1,
            )
        if bert_pooling_config.name == "concat-mean":
            self.pool_fc = nn.Linear(
                self.bert_model.config.hidden_size
                * len(bert_pooling_config.params["hs_idxs"]),
                self.bert_model.config.hidden_size,
            )

        self.dropout = nn.Dropout(p=model_config.dropout)
        self.fc = nn.Linear(
            self.bert_model.config.hidden_size, model_config.channel_size
        )
        self.bn = nn.BatchNorm1d(model_config.channel_size)

        if met_config.name == "ArcAdaptiveMarginProduct":
            self.margin = MetricLearningFactory.get_metric_learning_product(
                met_config,
                in_features=model_config.channel_size,
                out_features=out_features,
                train_df=train_df,
            )
        else:
            self.margin = MetricLearningFactory.get_metric_learning_product(
                met_config,
                in_features=model_config.channel_size,
                out_features=out_features,
            )

        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.bert_pooling_config.name == "mean":
            x = BertPoolingFactory.mean_pooling(output, attention_mask)
        elif self.bert_pooling_config.name == "concat-mean":
            hs = output.hidden_states
            hs_idxs = self.bert_pooling_config.params["hs_idxs"]
            seq_output = torch.cat([hs[idx] for idx in hs_idxs], dim=-1)
            avg_output = torch.sum(
                seq_output * attention_mask.unsqueeze(-1), dim=1, keepdim=False
            )
            avg_output = avg_output / torch.sum(attention_mask, dim=-1, keepdim=True)
            x = self.pool_fc(avg_output)
        elif self.bert_pooling_config.name == "concat-conv":
            hs = output.hidden_states
            hs_idxs = self.bert_pooling_config.params["hs_idxs"]
            seq_output = torch.cat([hs[idx] for idx in hs_idxs], dim=-1)
            x = self.conv(seq_output.permute(0, 2, 1)).permute(0, 2, 1).mean(1)
        elif self.bert_pooling_config.name == "hs-mean":
            hs = output.hidden_states
            hs_idxs = self.bert_pooling_config.params["hs_idxs"]
            seq_output = torch.stack([hs[idx] for idx in hs_idxs]).mean(dim=0)
            avg_output = torch.sum(
                seq_output * attention_mask.unsqueeze(-1), dim=1, keepdim=False
            )
            avg_output = avg_output / torch.sum(attention_mask, dim=-1, keepdim=True)
            x = avg_output
        elif self.bert_pooling_config.name == "naive_mean":
            x = BertPoolingFactory.naive_mean_pooling(
                output,
                attention_mask,
                last_layer_num=self.bert_pooling_config.params["last_layer_num"],
            )
        elif self.bert_pooling_config.name == "max":
            x = BertPoolingFactory.max_pooling(output, attention_mask)
        elif self.bert_pooling_config.name == "pooler":
            assert "pooler_output" in output, "pooler_output doesn't exists in output"
            x = output.pooler_output
        else:
            x = output.last_hidden_state[:, 0, :]

        out = self.fc(x)
        out = self.bn(out)
        if self.model_config.normalize:
            out = F.normalize(out)
        out = out.float()
        x = x.float()
        if labels is not None:
            return self.margin(out, labels)
        return out, x
