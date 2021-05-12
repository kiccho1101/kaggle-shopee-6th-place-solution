from dataclasses import dataclass

import pandas as pd
from kaggle_shopee.factories.config_factory import Config
from kaggle_shopee.utils.file_util import FileUtil
from kaggle_shopee.utils.test_util import TestUtil


@dataclass
class Data:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    train_fold: pd.DataFrame


class DataFactory:
    @staticmethod
    def load_data(config: Config) -> Data:
        data = Data(
            train=FileUtil.read_csv(config.dir_config.data_dir / "train.csv"),
            test=FileUtil.read_csv(config.dir_config.data_dir / "test.csv"),
            sample_submission=FileUtil.read_csv(
                config.dir_config.data_dir / "sample_submission.csv"
            ),
            train_fold=FileUtil.read_csv(config.dir_config.train_fold_dir)
        )
        return data
