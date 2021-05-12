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
from kaggle_shopee.factories.config_factory import (
    Config,
    ConfigFactory,
    EnvEnum,
    ReducerEnum,
)
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
    def main(data: Data, config: Config) -> Tuple[Data, Config]:
        data, config = Pp.image_path(data, config)
        return data, config


class FE:
    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def title_tfidf_reduced(data: Data, config: Config) -> Tuple[Data, Config]:
        tfidf = TfidfVectorizer(min_df=10)
        _df = pd.concat([data.train, data.test], axis=0).reset_index(drop=True)
        tfidf.fit(_df["title"])
        _train_encoded = tfidf.transform(data.train["title"]).todense()
        _test_encoded = tfidf.transform(data.test["title"]).todense()
        _encoded = np.concatenate([_train_encoded, _test_encoded], axis=0)
        TestUtil.assert_any(_train_encoded.shape[1], _test_encoded.shape[1])
        TestUtil.assert_any(_train_encoded.shape[1], _encoded.shape[1])

        if config.title_tfidf_reducer == ReducerEnum.NOTHING:
            config.title_tfidf_n_components = _encoded.shape[1]
            for i in tqdm(range(config.title_tfidf_n_components)):
                data.train[f"title_tfidf_{i}"] = _train_encoded[:, i]
                data.test[f"title_tfidf_{i}"] = _test_encoded[:, i]
            return data, config

        if config.title_tfidf_reducer == ReducerEnum.PCA:
            reduce_model = PCA(n_components=config.title_tfidf_n_components)
        else:
            reduce_model = PCA(n_components=config.title_tfidf_n_components)

        reduce_model.fit(_encoded)
        _train_reduced = reduce_model.transform(_train_encoded)
        _test_reduced = reduce_model.transform(_test_encoded)
        TestUtil.assert_any(_train_reduced.shape[1], _test_reduced.shape[1])
        TestUtil.assert_any(config.title_tfidf_n_components, _train_reduced.shape[1])
        for i in tqdm(range(config.title_tfidf_n_components)):
            data.train[f"title_tfidf_{i}"] = _train_reduced[:, i]
            data.test[f"title_tfidf_{i}"] = _test_reduced[:, i]
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    def main(data: Data, config: Config) -> Tuple[Data, Config]:
        data, config = FE.title_tfidf_reduced(data, config)
        return data, config


args = ArgsUtil.get_args(env=EnvEnum.LOCAL, exp="exp001")

yaml_str = """
    exp: exp001
    seed: 77

    title_tfidf_n_components: 50
    title_tfidf_reducer: PCA
"""

config = ConfigFactory.get_config_from_yaml_str(yaml_str)
data = DataFactory.load_data(config)
data, config = Pp.main(data, config)

# %%
ImgUtil.show_img(data.train["image_path"].iloc[2])

# %%
data, config = FE.main(data, config)


# %%
nbrs = NearestNeighbors(n_neighbors=5, algorithm="ball_tree")
x_train = data.train[
    [f"title_tfidf_{i}" for i in range(config.title_tfidf_n_components)]
].values
nbrs.fit(x_train)
distances, indices = nbrs.kneighbors(x_train)

# %%
data, config = Pp.main(data, config)

# %%
_df = data.train.groupby("label_group").posting_id.agg("unique").to_dict()
data.train["y_true"] = data.train["label_group"].map(_df)


_df = data.train.groupby("image_phash")["posting_id"].agg("unique").to_dict()
data.train["y_pred_phash"] = data.train["image_phash"].map(_df)
