import kaggle_shopee.cirtorch.layers.pooling as LP
import torch
import torch.nn as nn
from kaggle_shopee.factories.config_factory import PoolingConfig


class PoolingFactory:
    @staticmethod
    def get_pooling(config: PoolingConfig):
        poolings = {
            "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
            "MAC": LP.MAC,
            "SPoC": LP.SPoC,
            "GeM": LP.GeM,
            "GeMmp": LP.GeMmp,
            "RMAC": LP.RMAC,
            "Rpool": LP.Rpool,
        }
        assert config.name in poolings, "pooling_name not in {}".format(
            list(poolings.keys())
        )
        return poolings[config.name](**config.params)


class BertPoolingFactory:
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def naive_mean_pooling(model_output, attention_mask, last_layer_num: int = 128):
        token_embeddings = model_output[0]
        return torch.mean(token_embeddings[:, -last_layer_num:, :], 1)

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        max_embeddings = torch.max(token_embeddings * input_mask_expanded, 1)
        return max_embeddings
