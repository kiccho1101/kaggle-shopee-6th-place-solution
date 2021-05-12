import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from kaggle_shopee.factories.config_factory import (
    DatasetConfig,
    DirConfig,
    ModelConfig,
    TrainConfig,
)
from kaggle_shopee.factories.mining_factory import MiningFactory
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ShopeeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        output_label: bool,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        dir_config: DirConfig,
        transforms=None,
        tokenizer_path: Optional[str] = None,
        fold: Optional[int] = None,
    ):
        super().__init__()
        self.image_paths = df["image_path"].values
        self.output_label = output_label
        self.transforms = transforms
        self.posting_id_to_index = (
            df.reset_index(drop=True)
            .reset_index()
            .set_index("posting_id")["index"]
            .to_dict()
        )
        self.posting_ids = df["posting_id"].values

        _tokenizer_path = (
            model_config.bert_model_arch if tokenizer_path is None else tokenizer_path
        )
        print("tokenizer:", _tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path)
        texts = df[dataset_config.text_col].fillna("NaN").tolist()
        self.encodings = tokenizer(
            texts,
            padding=dataset_config.padding,
            truncation=dataset_config.truncation,
            max_length=dataset_config.max_length,
        )

        if output_label:
            self.labels = df[train_config.target].values

        self.with_triplets = False
        if output_label and dataset_config.is_with_triplets() and fold is not None:
            self.positive_dict, self.negative_dict = MiningFactory.get_triplets(
                dir_config,
                dataset_config.mining_exp,
                dataset_config.mining_epoch,
                fold,
                dataset_config.num_negatives,
            )
            self.with_triplets = True

    def __len__(self):
        return len(self.image_paths)

    def _get_item(self, index):
        image_path = self.image_paths[index]
        img = cv2.imread(image_path)
        assert img is not None, f"path: {image_path} doesn't exist"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        items = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}

        label = []
        if self.output_label:
            label = self.labels[index]
        return img, label, image_path, items

    def __getitem__(self, index: int):
        img, label, image_path, items = self._get_item(index)

        out = {"img": img, "label": label, "image_path": image_path, **items}
        if self.with_triplets:
            posting_id = self.posting_ids[index]
            np.random.seed(int(time.time()))
            p_posting_id = np.random.choice(self.positive_dict[posting_id], 1)[0]
            n_posting_id = np.random.choice(self.negative_dict[posting_id], 1)[0]
            p_index = self.posting_id_to_index[p_posting_id]
            n_index = self.posting_id_to_index[n_posting_id]
            positive_img, _, _, positive_items = self._get_item(p_index)
            negative_img, _, _, negative_items = self._get_item(n_index)
            out["positive"] = {"img": positive_img, **positive_items}
            out["negative"] = {"img": negative_img, **negative_items}

        return out
