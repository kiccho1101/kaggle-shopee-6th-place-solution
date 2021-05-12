import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import gc
import time
from typing import List

import cupy
import numpy as np
import pandas as pd
import torch
import torch.cuda
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.string_util import StringUtil
from kaggle_shopee.utils.test_util import TestUtil
from tqdm import tqdm

start = time.time()

args = ArgsUtil.get_args()
print(args)
config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, True)
data = DataFactory.load_data(config)

is_commiting = True if len(data.test) == 3 else False
sampling_ratio = 100
if is_commiting:
    data.test = pd.DataFrame(
        data.test.values.tolist() * int(70000 / len(data.test) / sampling_ratio),
        columns=data.test.columns,
    )

data, config = Pp.main(data, config)


def load_model(model_checkpoint: str):
    if args.env == EnvEnum.KAGGLE:
        model_checkpoint = model_checkpoint.replace("=", "")
    _exp = model_checkpoint.split("_")[0]
    _config = ConfigFactory.get_config_from_yaml_file(_exp, args.env, False)
    _config.model_config.pretrained = False
    _checkpoint_path = str(config.dir_config.checkpoint_dir / model_checkpoint)
    print("checkpoint:", _checkpoint_path)
    lit_model: lit_models.ShopeeLitModel = (
        lit_models.ShopeeLitModel.load_from_checkpoint(
            _checkpoint_path,
            data=data,
            config=_config,
            fold=-1,
            with_mlflow=False,
            bert_path=str(config.dir_config.input_dir / "kaggle-shopee-dataset"),
        )
    )
    model_type = StringUtil.get_model_type(_config.model_config.model_name)
    model = lit_model.model
    del lit_model
    gc.collect()
    return model, model_type


def get_features(model_checkpoint: str) -> np.ndarray:
    torch.cuda.empty_cache()
    test_dataloader = DataLoaderFactory.get_test_dataloader(data, config)

    device = "cuda"
    model, model_type = load_model(model_checkpoint)
    model.to(device)
    model.eval()

    features = []
    for batch in tqdm(test_dataloader):
        img = batch["img"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if model_type == "image+text":
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    features += [model(img, input_ids, attention_mask).detach().cpu()]
        elif model_type == "image":
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    features += [model(img).detach().cpu()]
        else:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    features += [model(input_ids, attention_mask).detach().cpu()]

    del model
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()
    features = torch.cat(features).cpu().numpy()
    return features


def get_batch_idxs(n_rows: int, n_batches: int):
    bs = n_rows // n_batches
    batch_idxs = []
    for i in range(n_batches):
        left = bs * i
        right = bs * (i + 1)
        if i == n_batches - 1:
            right = n_rows
        batch_idxs.append((left, right))
    return batch_idxs


def get_y_pred(
    features: np.ndarray, batch_idxs: List, posting_ids: np.ndarray, threshold: int
) -> List[List[str]]:
    y_pred: List[List[str]] = []
    for (left, right) in tqdm(batch_idxs):
        selection = (
            cupy.matmul(cupy.array(features[left:right, :]), cupy.array(features.T))
            > threshold / 100
        )
        for i in range(len(selection)):
            y_pred.append(posting_ids[cupy.asnumpy(selection[i])].tolist())
        torch.cuda.empty_cache()
    return y_pred


def inference(
    data: Data, features: np.ndarray, n_batches: int, threshold: int
) -> List[List[str]]:
    posting_ids = data.test["posting_id"].values
    n_rows = features.shape[0]
    batch_idxs = get_batch_idxs(n_rows, n_batches)
    y_pred = get_y_pred(features, batch_idxs, posting_ids, threshold)
    return y_pred


def get_tfidf_y_pred(n_batches: int, threshold: int) -> List[List[str]]:
    torch.cuda.empty_cache()
    model = TfidfVectorizer(stop_words="english", binary=True, max_features=10000)
    features = model.fit_transform(data.test["title"])
    posting_ids = data.test["posting_id"].values
    n_rows = features.shape[0]
    batch_idxs = get_batch_idxs(n_rows, n_batches)
    y_pred = get_y_pred(features, batch_idxs, posting_ids, threshold)
    del model
    del features
    gc.collect()
    torch.cuda.empty_cache()
    return y_pred


y_pred_df = pd.DataFrame()
for i in range(len(config.inference_config.model_checkpoints)):
    features = get_features(config.inference_config.model_checkpoints[i])
    y_pred = inference(data, features, 20, config.inference_config.thresholds[i])
    y_pred_df[f"y_pred_{i}"] = y_pred

y_pred_df["y_pred_tfidf"] = get_tfidf_y_pred(20, 85)

if is_commiting:
    end = time.time()
    print("Estimated inference time: {} seconds.", int((end - start) * sampling_ratio))

if not is_commiting:
    data.sample_submission["matches"] = y_pred_df.apply(
        lambda row: " ".join(np.unique(np.concatenate(row))), axis=1
    )
data.sample_submission.to_csv("submission.csv", index=False)
