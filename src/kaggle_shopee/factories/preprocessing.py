import re
from typing import Tuple

from kaggle_shopee.factories.config_factory import Config, EnvEnum
from kaggle_shopee.factories.data_factory import Data
from kaggle_shopee.utils.test_util import TestUtil
from kaggle_shopee.utils.time_util import TimeUtil
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder


def num_to_str(n):
    a = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    b = [
        "ten",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]
    c = [
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]

    # 1000の位の計算
    dd = n // 1000

    # 100の位の計算(nを100で割った整数を求める)
    d = n // 100
    # 10の位の計算(nからd * 100の値を引いた数を10で割った整数を求める)
    e = (n - d * 100) // 10
    # 1の位の計算(nからd * 100の値を引いた数を10で割った余りを求める)
    f = (n - d * 100) % 10

    if dd >= 1:
        return str(n)

    if n >= 11 and n <= 19:  # nが11~19の数かを調べる
        return c[n - 11]
    elif d == 0:  # 100の位があるか調べる
        if e == 0:  # 10の位があるか調べる
            return a[f - 1]
        elif f == 0:  # 1の位があるか調べる
            return b[e - 1]
        else:
            return b[e - 1] + "-" + a[f - 1]
    elif e == 0:
        if f == 0:
            return a[d - 1] + " hundred"
        else:
            return a[d - 1] + " hundred and " + a[f - 1]
    else:
        if f == 0:
            return a[d - 1] + " hundred and " + b[e - 1]
        else:
            return a[d - 1] + " hundred and " + b[e - 1] + "-" + a[f - 1]


def title_number_to_str(df):
    regex = re.compile(r"\d+")
    title = df["title"].values

    for row in range(len(title)):
        text = title[row]
        odds = []

        for line in text.splitlines():
            match = regex.findall(line)

            for i in match:
                num_str = num_to_str(int(i))
                odds.append(num_str)

            for j, k in zip(match, odds):
                title[row] = df["title"][row].replace(j, k)

    df["title_num_str"] = title

    return df


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
    @TestUtil.test_data
    def split_folds(data: Data, config: Config) -> Tuple[Data, Config]:
        folds = GroupKFold(n_splits=config.cv_config.n_splits)
        data.train["fold"] = -1
        for fold, (_, valid_idx) in enumerate(
            folds.split(data.train, None, data.train["label_group"])
        ):
            data.train.loc[valid_idx, "fold"] = fold
        data.train["kurupical_fold"] = data.train["fold"]
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def kurupical_fold(data: Data, config: Config) -> Tuple[Data, Config]:
        # if config.env != EnvEnum.KAGGLE:
        #     data.train = data.train.merge(
        #         data.train_fold[["posting_id", "fold"]].rename(
        #             {"fold": "kurupical_fold"}, axis=1
        #         ),
        #         on="posting_id",
        #     )
        # else:
        #     data.train["kurupical_fold"] = data.train["fold"]
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def label_group_le(data: Data, config: Config) -> Tuple[Data, Config]:
        le = LabelEncoder()
        data.train["label_group_le"] = le.fit_transform(data.train["label_group"])
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def target(data: Data, config: Config) -> Tuple[Data, Config]:
        tmp = data.train.groupby("label_group")["posting_id"].unique()
        data.train["target"] = data.train["label_group"].map(tmp)
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def title_preprocessed(data: Data, config: Config) -> Tuple[Data, Config]:
        def string_escape(s, encoding="utf-8"):
            return (
                s.encode("latin1")  # To bytes, required by 'unicode-escape'
                .decode("unicode-escape")  # Perform the actual octal-escaping decode
                .encode("latin1")  # 1:1 mapping back to bytes
                .decode(encoding)
            )  # Decode original encoding

        data.train["title_preprocessed"] = data.train["title"].map(string_escape)
        data.test["title_preprocessed"] = data.test["title"].map(string_escape)
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def title_num_str(data: Data, config: Config) -> Tuple[Data, Config]:
        data.train = title_number_to_str(data.train)
        data.test = title_number_to_str(data.test)
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    @TestUtil.test_data
    def shuffle(data: Data, config: Config) -> Tuple[Data, Config]:
        data.train = data.train.sample(frac=1, random_state=config.seed).reset_index(
            drop=True
        )
        return data, config

    @staticmethod
    @TimeUtil.timer_wrapper
    def main(data: Data, config: Config) -> Tuple[Data, Config]:
        data, config = Pp.image_path(data, config)
        data, config = Pp.split_folds(data, config)
        data, config = Pp.kurupical_fold(data, config)
        data, config = Pp.label_group_le(data, config)
        data, config = Pp.target(data, config)
        data, config = Pp.title_preprocessed(data, config)
        data, config = Pp.title_num_str(data, config)
        data, config = Pp.shuffle(data, config)
        return data, config
