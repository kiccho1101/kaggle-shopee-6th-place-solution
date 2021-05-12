from typing import Tuple

from kaggle_shopee.factories.config_factory import Config
from kaggle_shopee.factories.data_factory import Data
from kaggle_shopee.utils.test_util import TestUtil
from kaggle_shopee.utils.time_util import TimeUtil


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
