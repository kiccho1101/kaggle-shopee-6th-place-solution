import sys
from pathlib import Path

try:
    sys.path.append(str(Path(__file__).parents[2]))
except Exception:
    pass
sys.path.append("/kaggle/input/kaggle-shopee/src")

import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.cuda
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import (Config, ConfigFactory,
                                                    EnvEnum)
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from tqdm import tqdm


def get_kiccho_embeddings(
    exp: str,
    test_df: pd.DataFrame,
    num_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    image_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    input
        exp: 実験名。expxxx.yamlのinference_configに使うモデルの情報を書く
        test_df: test.csvをpd.read_csvで読んだもの

    output
        all_embeddings, img_embeddings, text_embeddings
    """
    args = ArgsUtil.get_args(EnvEnum.KAGGLE, exp, [])

    device = "cuda"

    print(args)
    config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, verbose=False)
    data = DataFactory.load_data(config)

    data.test = test_df.copy()
    data, config = Pp.image_path(data, config)
    data, config = Pp.label_group_le(data, config)
    data, config = Pp.split_folds(data, config)
    data, config = Pp.kurupical_fold(data, config)

    if image_dir is not None:
        data.test["image_path"] = data.test["image"].map(lambda i: f"{image_dir}/{i}")

    model_checkpoint = config.inference_config.epoch_configs[0].model_checkpoints[0]
    if args.env == EnvEnum.KAGGLE:
        model_checkpoint = model_checkpoint.replace("=", "")

    print("load model:", model_checkpoint)
    model = lit_models.ShopeeLitModel.load_from_checkpoint(
        str(config.dir_config.checkpoint_dir / model_checkpoint),
        data=data,
        config=config,
        fold=-1,
        with_mlflow=False,
        bert_path=str(
            config.dir_config.dataset_dir / config.model_config.bert_model_arch
        ),
        is_test=True,
    ).model.to(device)
    model.eval()

    test_dataloader = DataLoaderFactory.get_test_dataloader(
        data, config, num_workers=num_workers, batch_size=batch_size
    )

    img_features = []
    text_features = []
    all_features = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            img = batch["img"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            all_feature, img_feature, text_feature = model(
                img, input_ids, attention_mask
            )

            all_features.extend(all_feature.detach().cpu().numpy().astype(np.float16))
            img_features.extend(img_feature.detach().cpu().numpy().astype(np.float16))
            text_features.extend(text_feature.detach().cpu().numpy().astype(np.float16))

    img_features = np.array(img_features, dtype=np.float16)
    text_features = np.array(text_features, dtype=np.float16)
    all_features = np.array(all_features, dtype=np.float16)

    del data
    del model
    del test_dataloader.dataset
    del test_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    return all_features, img_features, text_features
