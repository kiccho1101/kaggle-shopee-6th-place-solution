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
from typing import Any, Dict, List, Optional, Tuple

import cudf
import cupy
import numpy as np
import pandas as pd
import torch
import torch.cuda
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import Config, ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.inference_factory import InferenceFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.string_util import StringUtil
from kaggle_shopee.utils.test_util import TestUtil
from sklearn.preprocessing import normalize
from tqdm import tqdm


def get_kiccho_embeddings(
    exp: str,
    test_df: pd.DataFrame,
    num_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    image_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """
    input
        exp: 実験名。expxxx.yamlのinference_configに使うモデルの情報を書く
        test_df: test.csvをpd.read_csvで読んだもの

    output
        embeddings: Dict[str, np.ndarray]
            key: 実験名 (exp001とか)
            value: embedding. shapeは ( len(test_df) x linear_out )
    """
    args = ArgsUtil.get_args(EnvEnum.KAGGLE, exp, [])

    print(args)
    config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, True)
    data = DataFactory.load_data(config)

    data.test = test_df
    data, config = Pp.image_path(data, config)
    data, config = Pp.label_group_le(data, config)
    data, config = Pp.split_folds(data, config)
    data, config = Pp.kurupical_fold(data, config)

    if image_dir is not None:
        data.test["image_path"] = data.test["image"].map(lambda i: f"{image_dir}/{i}")

    features = []
    img_features = []
    txt_features = []
    for epoch_config in config.inference_config.epoch_configs:
        _config = ConfigFactory.get_config_from_yaml_file(
            epoch_config.dataloader_exp, args.env, False
        )
        test_dataloader = DataLoaderFactory.get_test_dataloader(
            data, _config, num_workers=num_workers, batch_size=batch_size
        )
        _features, _img_features, _txt_features = InferenceFactory.epoch(
            args.env, epoch_config, test_dataloader, data
        )
        features += _features
        img_features += _img_features
        txt_features += _txt_features
        del _features
        del _img_features
        del _txt_features
        gc.collect()
    for i in range(len(features)):
        features[i] = np.concatenate(features[i])
        img_features[i] = np.concatenate(img_features[i])
        txt_features[i] = np.concatenate(txt_features[i])
        print(f"features[{i}].shape:", features[i].shape)
        print(f"img_features[{i}].shape:", img_features[i].shape)
        print(f"txt_features[{i}].shape:", txt_features[i].shape)

    exps: List[str] = []
    for epoch_config in config.inference_config.epoch_configs:
        for model_checkpoint in epoch_config.model_checkpoints:
            _exp = model_checkpoint.split("_")[0]
            exps.append(_exp)

    TestUtil.assert_any(len(exps), len(features))

    return features[0], img_features[0], txt_features[0]
