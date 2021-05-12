import sys
from pathlib import Path

try:
    sys.path.append(str(Path(__file__).parents[2]))
except Exception:
    pass
sys.path.append("/kaggle/input/kaggle-shopee/src")

import copy
import gc
import time
from typing import Any, List, Tuple

import cudf
import cupy
import numpy as np
import pandas as pd
import torch
import torch.cuda
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import (Config, ConfigFactory,
                                                    EnvEnum)
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.inference_factory import InferenceFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.string_util import StringUtil
from kaggle_shopee.utils.test_util import TestUtil
from sklearn.preprocessing import normalize
from tqdm import tqdm

start = time.time()

args = ArgsUtil.get_args()
# args = ArgsUtil.get_args(EnvEnum.KAGGLE, "exp017", [])

print(args)
config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, True)
data = DataFactory.load_data(config)

is_commiting = len(data.test) == 3
sampling_ratio = 100
if is_commiting:
    data.test = pd.DataFrame(
        data.test.values.tolist() * int(70000 / len(data.test) / sampling_ratio),
        columns=data.test.columns,
    )

data, config = Pp.main(data, config)
# data.test = pd.concat([data.train, data.train], axis=0).reset_index(drop=True)

features = []
posting_ids = data.test["posting_id"].values
for epoch_config in config.inference_config.epoch_configs:
    _config = ConfigFactory.get_config_from_yaml_file(
        epoch_config.dataloader_exp, args.env, False
    )
    test_dataloader = DataLoaderFactory.get_test_dataloader(data, _config)
    features += InferenceFactory.epoch(args.env, epoch_config, test_dataloader, data)
for i in range(len(features)):
    features[i] = torch.cat(features[i]).cpu().numpy()
    print(f"features[{i}].shape:", features[i].shape)
__features = []
for concat_config in config.inference_config.concat_configs:
    _features = copy.deepcopy(features)
    for idx, weight in zip(concat_config.idxs, concat_config.weights):
        _features[idx] = _features[idx] * weight / np.sum(concat_config.weights)
    _features = np.concatenate([_features[idx] for idx in concat_config.idxs], axis=1)
    _features = normalize(_features)
    __features.append(_features)
for i in range(len(__features)):
    print(f"__features[{i}].shape:", __features[i].shape)
batch_idxs = InferenceFactory.get_batch_idxs(len(data.test), n_batches=20)
y_pred_df = InferenceFactory.get_inference_y_pred_df(
    batch_idxs, __features, config.inference_config.thresholds, posting_ids
)

if not is_commiting:
    data.sample_submission["matches"] = y_pred_df.apply(
        lambda row: " ".join(np.unique(np.concatenate(row))), axis=1
    )
data.sample_submission.to_csv("submission.csv", index=False)
