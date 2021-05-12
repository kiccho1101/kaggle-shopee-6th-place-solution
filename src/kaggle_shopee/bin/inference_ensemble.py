import sys
from pathlib import Path

try:
    sys.path.append(str(Path(__file__).parents[2]))
except Exception:
    pass
sys.path.append("/kaggle/input/kaggle-shopee/src")

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
from kaggle_shopee.factories.config_factory import Config, ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.inference_factory import InferenceFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.string_util import StringUtil
from kaggle_shopee.utils.test_util import TestUtil
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


def load_config(model_checkpoint: str) -> Config:
    _exp = model_checkpoint.split("_")[0]
    _config = ConfigFactory.get_config_from_yaml_file(_exp, args.env, False)
    _config.model_config.pretrained = False
    return _config


def load_model(model_checkpoint: str) -> Tuple[Any, str, Config]:
    if args.env == EnvEnum.KAGGLE:
        model_checkpoint = model_checkpoint.replace("=", "")
    _config = load_config(model_checkpoint)
    lit_model: lit_models.ShopeeLitModel = (
        lit_models.ShopeeLitModel.load_from_checkpoint(
            str(config.dir_config.checkpoint_dir / model_checkpoint),
            data=data,
            config=_config,
            fold=-1,
            with_mlflow=False,
            bert_path=str(
                config.dir_config.dataset_dir / _config.model_config.bert_model_arch
            ),
        )
    )
    model_type = StringUtil.get_model_type(_config.model_config.model_name)
    model = lit_model.model
    model.to("cuda")
    model.eval()
    del lit_model
    gc.collect()
    return model, model_type, _config


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


def get_tfidf_y_pred(
    n_batches: int, threshold: int, max_features: int
) -> List[List[str]]:
    torch.cuda.empty_cache()
    model = TfidfVectorizer(
        stop_words="english", binary=True, max_features=max_features
    )
    _gf = cudf.from_pandas(data.test)
    features = model.fit_transform(_gf["title"]).toarray()
    del _gf
    gc.collect()
    posting_ids = data.test["posting_id"].values
    n_rows = features.shape[0]
    batch_idxs = get_batch_idxs(n_rows, n_batches)
    y_pred = get_y_pred(features, batch_idxs, posting_ids, threshold)
    del model
    del features
    gc.collect()
    torch.cuda.empty_cache()
    return y_pred


def epoch(model_checkpoints: List[str], data: Data):
    device = "cuda"
    models: List[Any] = []
    model_types: List[str] = []
    _features = [[] for _ in range(len(model_checkpoints))]
    _config = load_config(model_checkpoints[0])
    test_dataloader = DataLoaderFactory.get_test_dataloader(data, _config)
    for i in range(len(model_checkpoints)):
        model, model_type, _config = load_model(model_checkpoints[i])
        models.append(model)
        model_types.append(model_type)
    for i in range(len(models)):
        models[i].to(device)
    for batch in tqdm(test_dataloader):
        img = batch["img"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for i in range(len(models)):
                    if model_types[i] == "image+text":
                        _features[i] += [
                            models[i](img, input_ids, attention_mask).detach().cpu()
                        ]
                    elif model_types[i] == "image":
                        _features[i] += [models[i](img).detach().cpu()]
                    else:
                        _features[i] += [
                            models[i](input_ids, attention_mask).detach().cpu()
                        ]
    del models
    del test_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    return _features


device = "cuda"
TestUtil.assert_any(
    len(config.inference_config.thresholds),
    len(config.inference_config.model_checkpoints),
)
features = []
features += epoch(config.inference_config.model_checkpoints[:2], data)
if len(config.inference_config.model_checkpoints) > 2:
    features += epoch(config.inference_config.model_checkpoints[2:], data)

for i in range(len(features)):
    features[i] = torch.cat(features[i]).cpu().numpy()

TestUtil.assert_any(len(features), len(config.inference_config.model_checkpoints))
y_pred = [[] for _ in range(len(features))]
posting_ids = data.test["posting_id"].values
n_rows = len(data.test)
n_batches = 20
batch_idxs = get_batch_idxs(n_rows, n_batches)
for (left, right) in tqdm(batch_idxs):
    for i in range(len(features)):
        selection = (
            cupy.matmul(
                cupy.array(features[i][left:right, :]), cupy.array(features[i].T)
            )
            > config.inference_config.thresholds[i] / 100
        )
        for j in range(len(selection)):
            y_pred[i].append(posting_ids[cupy.asnumpy(selection[j])].tolist())
        torch.cuda.empty_cache()

y_pred_df = pd.DataFrame()
for i in range(len(y_pred)):
    y_pred_df[f"y_pred_{i}"] = y_pred[i]

if config.inference_config.with_tfidf:
    y_pred_df["y_pred_tfidf"] = get_tfidf_y_pred(
        20,
        config.inference_config.tfidf_threshold,
        config.inference_config.tfidf_max_features,
    )

if is_commiting:
    end = time.time()
    print("Estimated inference time: {} seconds.", int((end - start) * sampling_ratio))

if not is_commiting:
    data.sample_submission["matches"] = y_pred_df.apply(
        lambda row: " ".join(np.unique(np.concatenate(row))), axis=1
    )
    # y_pred_df["y_pred"] = y_pred_df.apply(
    #     lambda row: np.unique(np.concatenate(row)).tolist(), axis=1
    # )
data.sample_submission.to_csv("submission.csv", index=False)
