import gc
from collections import defaultdict
from typing import Dict, List, Tuple

import cupy
import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.nn.functional as F
from kaggle_shopee.factories.config_factory import ConfigFactory, DirConfig, EnvEnum
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.string_util import StringUtil
from tqdm.autonotebook import tqdm


def get_mining_features(
    dir_config: DirConfig, offline_mining_exp: str, epoch: int, fold: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    df = pd.DataFrame()
    features = []
    epoch_str = str(epoch).zfill(2)
    mining_dir = dir_config.output_dir.parent / offline_mining_exp
    print("loading mining features...")
    for _fold in tqdm(range(5)):
        if _fold != fold:
            _df = pd.read_csv(mining_dir / f"valid_df_{_fold}.csv")
            _df["target"] = _df["target"].map(StringUtil.str_to_list)
            _features = np.load(mining_dir / f"features_{_fold}_{epoch_str}.npy")
            df = pd.concat([df, _df], axis=0)
            features.append(_features)
    features = np.concatenate(features)
    features = F.normalize(torch.from_numpy(features)).detach().numpy()
    df = df.reset_index(drop=True)
    return features, df


def get_batch_idxs(n_rows: int, n_batches: int = 20) -> List[Tuple[int, int]]:
    bs = n_rows // n_batches
    batch_idxs = []
    for i in range(n_batches):
        left = bs * i
        right = bs * (i + 1)
        if i == n_batches - 1:
            right = n_rows
        batch_idxs.append((left, right))
    return batch_idxs


def get_positives(train_df: pd.DataFrame) -> Tuple[List[List[str]], defaultdict]:
    print("getting positives...")
    positives: List[List[str]] = []
    positive_dict = defaultdict(lambda: [])
    for row in tqdm(
        train_df[["posting_id", "target"]].itertuples(), total=len(train_df)
    ):
        for target in row.target:
            if target != row.posting_id:
                positives.append([row.posting_id, target])
                positive_dict[row.posting_id].append(target)
    return positives, positive_dict


def get_negatives(
    posting_ids: np.ndarray,
    features,
    batch_idxs,
    positive_dict,
    num_negatives: int = 3,
) -> List[List[str]]:
    print("getting negatives...")
    negatives: List[List[str]] = []
    for (left, right) in tqdm(batch_idxs):
        dot_product = cupy.asnumpy(
            cupy.matmul(
                cupy.array(features[left:right]),
                cupy.array(features.T),
            )
        )
        indices = np.argsort(dot_product)
        for i in range(len(indices)):
            posting_id = posting_ids[i + left]
            num_ids = 0
            for j in range(len(indices[i])):
                n_posting_id = posting_ids[indices[i][-j]]
                if (
                    n_posting_id != posting_id
                    and n_posting_id not in positive_dict[posting_id]
                ):
                    negatives.append([posting_id, n_posting_id])
                    num_ids += 1
                    if num_ids > num_negatives:
                        break
        del dot_product
        del indices
        gc.collect()
        torch.cuda.empty_cache()
    return negatives


class MiningFactory:
    @staticmethod
    def get_triplets(
        dir_config: DirConfig,
        offline_mining_exp: str,
        epoch: int,
        fold: int,
        num_negatives: int,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        features, train_df = get_mining_features(
            dir_config, offline_mining_exp, epoch, fold
        )
        posting_ids = train_df["posting_id"].values
        batch_idxs = get_batch_idxs(len(posting_ids), 20)
        positives, positive_dict = get_positives(train_df)
        negatives = get_negatives(
            posting_ids, features, batch_idxs, positive_dict, num_negatives
        )
        positive_df = pd.DataFrame(positives, columns=["posting_id", "p_posting_id"])
        negative_df = pd.DataFrame(negatives, columns=["posting_id", "n_posting_id"])
        positive_dict = (
            positive_df.groupby("posting_id")["p_posting_id"].unique().to_dict()
        )
        negative_dict = (
            negative_df.groupby("posting_id")["n_posting_id"].unique().to_dict()
        )
        del positive_df
        del negative_df
        return positive_dict, negative_dict


if __name__ == "__main__":
    offline_mining_exp = "exp373"
    epoch = 9
    fold = 0
    num_negatives = 3

    args = ArgsUtil.get_args(EnvEnum.COLAB, "exp383", [0])
    print(args)
    config = ConfigFactory.get_config_from_yaml_file(args.exp, env=args.env)

    positive_dict, negative_dict = MiningFactory.get_triplets(
        config.dir_config, offline_mining_exp, epoch, fold, num_negatives
    )
