import copy
import gc
import itertools
from typing import Any, Dict, List, Optional, Tuple

import cupy
import numpy as np
import pandas as pd
import torch
import torch.cuda
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import (Config, ConfigFactory,
                                                    EnvEnum,
                                                    InferenceConcatConfig,
                                                    InferenceConfig,
                                                    InferenceEpochConfig)
from kaggle_shopee.factories.data_factory import Data
from kaggle_shopee.utils.metric_util import MetricUtil
from kaggle_shopee.utils.string_util import StringUtil
from kaggle_shopee.utils.test_util import TestUtil
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AutoModel, AutoTokenizer


class InferenceFactory:
    @staticmethod
    def test_config(config: Config):
        n_checkpoits = np.sum(
            [len(e.model_checkpoints) for e in config.inference_config.epoch_configs]
        )
        TestUtil.assert_any(int(n_checkpoits), len(config.inference_config.thresholds))

    @staticmethod
    def prepare_bert(env: EnvEnum, exp: str):
        config = ConfigFactory.get_config_from_yaml_file(exp, env, False)
        bert_dir = config.dir_config.dataset_dir / config.model_config.bert_model_arch
        if not bert_dir.exists():
            print(bert_dir)
            bert_model = AutoModel.from_pretrained(config.model_config.bert_model_arch)
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_config.bert_model_arch
            )
            bert_model.save_pretrained(bert_dir)
            tokenizer.save_pretrained(bert_dir)

            # Test
            AutoModel.from_pretrained(bert_dir)
            AutoTokenizer.from_pretrained(bert_dir)

    @staticmethod
    def load_config(env: EnvEnum, model_checkpoint: str) -> Config:
        exp = model_checkpoint.split("_")[0]
        config = ConfigFactory.get_config_from_yaml_file(exp, env, False)
        config.model_config.pretrained = False
        return config

    @staticmethod
    def load_model(
        env: EnvEnum, model_checkpoint: str, data: Data, device: str = "cuda"
    ) -> Tuple[Any, str, Config, str]:
        if env == EnvEnum.KAGGLE:
            model_checkpoint = model_checkpoint.replace("=", "")
        config = InferenceFactory.load_config(env, model_checkpoint)
        print("load model:", model_checkpoint)
        lit_model: lit_models.ShopeeLitModel = (
            lit_models.ShopeeLitModel.load_from_checkpoint(
                str(config.dir_config.checkpoint_dir / model_checkpoint),
                data=data,
                config=config,
                fold=-1,
                with_mlflow=False,
                bert_path=str(
                    config.dir_config.dataset_dir / config.model_config.bert_model_arch
                ),
                is_test=True,
            )
        )
        model_type = StringUtil.get_model_type(config.model_config.model_name)
        model = lit_model.model
        model.to(device)
        model.eval()
        del lit_model
        gc.collect()
        return model, model_type, config, config.model_config.model_name

    @staticmethod
    def epoch(
        env: EnvEnum,
        epoch_config: InferenceEpochConfig,
        dataloader: DataLoader,
        data: Data,
        device: str = "cuda",
    ) -> Tuple[List, List, List]:
        models: List[Any] = []
        model_types: List[str] = []
        model_names: List[str] = []
        features = [[] for _ in range(len(epoch_config.model_checkpoints))]
        img_features = [[] for _ in range(len(epoch_config.model_checkpoints))]
        txt_features = [[] for _ in range(len(epoch_config.model_checkpoints))]
        for i in range(len(epoch_config.model_checkpoints)):
            model, model_type, _, model_name = InferenceFactory.load_model(
                env, epoch_config.model_checkpoints[i], data, device
            )
            model.eval()
            model.to(device)
            models.append(model)
            model_names.append(model_name)
            model_types.append(model_type)
            del model
            gc.collect()
        for batch in tqdm(dataloader, total=len(dataloader)):
            img = batch["img"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for i in range(len(models)):
                        if model_names[i] in ["ShopeeImgTextNet4", "ShopeeImgTextNet6"]:
                            _feature, _img_feature, _txt_feature = models[i](
                                img, input_ids, attention_mask
                            )
                            _feature = (
                                _feature.detach().cpu().numpy().astype(np.float16)
                            )
                            _img_feature = (
                                _img_feature.detach().cpu().numpy().astype(np.float16)
                            )
                            _txt_feature = (
                                _txt_feature.detach().cpu().numpy().astype(np.float16)
                            )
                            features[i] += [_feature]
                            img_features[i] += [_img_feature]
                            txt_features[i] += [_txt_feature]
                            del _feature
                            del _img_feature
                            del _txt_feature
                        elif model_types[i] == "image+text":
                            features[i] += [
                                models[i](img, input_ids, attention_mask).detach().cpu()
                            ]
                        elif model_types[i] == "image":
                            features[i] += [models[i](img).detach().cpu()]
                        else:
                            features[i] += [
                                models[i](input_ids, attention_mask).detach().cpu()
                            ]
        del models
        gc.collect()
        torch.cuda.empty_cache()
        return features, img_features, txt_features

    @staticmethod
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

    @staticmethod
    def _calc_metric(
        valid_df: pd.DataFrame, _y_pred_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Any, Any, Any]:
        _valid_df = valid_df.reset_index(drop=True).copy()
        TestUtil.assert_any(len(_valid_df), len(_y_pred_df))
        f1_scores = MetricUtil.f1_scores(
            _valid_df["target"].tolist(), _y_pred_df["y_pred"].tolist()
        )
        precisions, recalls = MetricUtil.precision_recall(
            _valid_df["target"].tolist(), _y_pred_df["y_pred"].tolist()
        )
        _y_pred_df["f1_score"] = f1_scores
        _y_pred_df["precision"] = precisions
        _y_pred_df["recall"] = recalls
        f1_score, precision, recall = (
            np.mean(f1_scores),
            np.mean(precisions),
            np.mean(recalls),
        )
        return _y_pred_df, f1_score, precision, recall

    @staticmethod
    def get_y_pred_df(
        batch_idxs: List[Tuple[int, int]],
        features,
        thresholds: List[List[int]],
        posting_ids: np.ndarray,
        tqdm_disable: bool = False,
        valid_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Tuple]:
        TestUtil.assert_any(len(features), len(thresholds))
        best_y_pred_df = pd.DataFrame()
        best_score = 0
        best_thresholds = ()
        for th in itertools.product(*thresholds):
            _y_pred = [[] for _ in range(len(features))]
            for (left, right) in tqdm(batch_idxs, disable=tqdm_disable):
                for i in range(len(features)):
                    selection = cupy.asnumpy(
                        cupy.matmul(
                            cupy.array(features[i][left:right, :]),
                            cupy.array(features[i].T),
                        )
                        > th[i] / 100
                    )
                    for j in range(len(selection)):
                        _y_pred[i].append(posting_ids[selection[j]].tolist())
                    torch.cuda.empty_cache()
            _y_pred_df = pd.DataFrame()
            for i in range(len(_y_pred)):
                _y_pred_df[f"y_pred_{i}"] = _y_pred[i]
            if valid_df is None:
                best_y_pred_df = _y_pred_df
            else:
                _y_pred_df["y_pred"] = _y_pred_df.filter(like="y_pred_").apply(
                    lambda row: np.unique(np.concatenate(row)).tolist(), axis=1
                )
                _y_pred_df, f1_score, precision, recall = InferenceFactory._calc_metric(
                    valid_df, _y_pred_df
                )
                print(
                    f"----------- valid f1: {f1_score} best: {best_score} precision: {precision} recall: {recall} threshold: {th} ------------"
                )
                if best_score < f1_score:
                    best_score = f1_score
                    best_y_pred_df = _y_pred_df.copy()
                    best_thresholds = th
        return best_y_pred_df, best_thresholds

    @staticmethod
    def get_concat_y_pred_df(
        batch_idxs: List[Tuple[int, int]],
        features,
        thresholds: List[int],
        posting_ids: np.ndarray,
        weights_list: List[List[int]],
        tqdm_disable: bool = False,
        valid_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, int, Tuple]:
        TestUtil.assert_any(len(features), len(weights_list))
        best_y_pred_df = pd.DataFrame()
        best_score = 0
        best_threshold = 0
        best_weights = ()
        for weights in itertools.product(*weights_list):
            _features = copy.deepcopy(features)
            for i in range(len(_features)):
                _features[i] = _features[i] * weights[i] / np.sum(weights)
            _features = np.concatenate(_features, axis=1)
            _features = normalize(_features)
            _best_y_pred_df = pd.DataFrame()
            _best_score = 0
            _best_threshold = 0
            for th in thresholds:
                _y_pred = []
                for (left, right) in tqdm(batch_idxs, disable=tqdm_disable):
                    selection = cupy.asnumpy(
                        cupy.matmul(
                            cupy.array(_features[left:right, :]),
                            cupy.array(_features.T),
                        )
                        > th / 100
                    )
                    for j in range(len(selection)):
                        _y_pred.append(posting_ids[selection[j]].tolist())
                    torch.cuda.empty_cache()
                _y_pred_df = pd.DataFrame()
                _y_pred_df["y_pred"] = _y_pred
                if valid_df is None:
                    best_y_pred_df = _y_pred_df
                else:
                    (
                        _y_pred_df,
                        f1_score,
                        precision,
                        recall,
                    ) = InferenceFactory._calc_metric(valid_df, _y_pred_df)
                    print(
                        f"----------- valid f1: {f1_score} best: {best_score} precision: {precision} recall: {recall} weights: {weights} threshold: {th} ------------"
                    )
                    if _best_score < f1_score:
                        _best_score = f1_score
                        _best_y_pred_df = _y_pred_df.copy()
                        _best_threshold = th
            if best_score < _best_score:
                best_score = _best_score
                best_y_pred_df = _best_y_pred_df
                best_weights = weights
                best_threshold = _best_threshold
        return best_y_pred_df, best_threshold, best_weights

    @staticmethod
    def get_inference_y_pred_df(
        batch_idxs: List[Tuple[int, int]],
        features,
        thresholds: List[int],
        posting_ids: np.ndarray,
        tqdm_disable: bool = False,
    ) -> pd.DataFrame:
        y_pred_df = pd.DataFrame()
        for i in range(len(features)):
            _y_pred = []
            for (left, right) in tqdm(batch_idxs, disable=tqdm_disable):
                selection = cupy.asnumpy(
                    cupy.matmul(
                        cupy.array(features[i][left:right, :]),
                        cupy.array(features[i].T),
                    )
                    > thresholds[i] / 100
                )
                for j in range(len(selection)):
                    _y_pred.append(posting_ids[selection[j]].tolist())
                torch.cuda.empty_cache()
            y_pred_df[f"y_pred_{i}"] = _y_pred
        return y_pred_df

    @staticmethod
    def get_avg_voting_inference_y_pred_df(
        batch_idxs: List[Tuple[int, int]],
        features,
        thresholds: List[int],
        concat_configs: List[InferenceConcatConfig],
        posting_ids: np.ndarray,
        tqdm_disable: bool = False,
    ) -> pd.DataFrame:
        y_pred_df = pd.DataFrame()
        TestUtil.assert_any(len(thresholds), len(concat_configs))
        _y_pred = [[] for _ in concat_configs]
        for i, concat_config in enumerate(concat_configs):
            for (left, right) in tqdm(batch_idxs, disable=tqdm_disable):
                dot_product = np.zeros(
                    (right - left, len(posting_ids)), dtype=np.float64
                )
                for idx, weight in zip(concat_config.idxs, concat_config.weights):
                    dot_product += (
                        weight
                        * cupy.asnumpy(
                            cupy.matmul(
                                cupy.array(features[idx][left:right, :]),
                                cupy.array(features[idx].T),
                            )
                        )
                        / np.sum(concat_config.weights)
                    )
                selection = dot_product > thresholds[i] / 100
                for j in range(len(selection)):
                    IDX = selection[j]
                    if np.sum(IDX) < concat_config.min_indices:
                        IDX = np.argsort(dot_product[j])[-concat_config.min_indices :]
                    _y_pred[i].append(posting_ids[IDX].tolist())
                torch.cuda.empty_cache()
        for i in range(len(_y_pred)):
            y_pred_df[f"y_pred_{i}"] = _y_pred[i]
        return y_pred_df

    @staticmethod
    def get_concat_inference_y_pred_df(
        batch_idxs: List[Tuple[int, int]],
        features: np.ndarray,
        threshold: int,
        posting_ids: np.ndarray,
        tqdm_disable: bool = False,
    ) -> pd.DataFrame:
        y_pred = []
        for (left, right) in tqdm(batch_idxs, disable=tqdm_disable):
            selection = cupy.asnumpy(
                cupy.matmul(
                    cupy.array(features[left:right, :]),
                    cupy.array(features.T),
                )
                > threshold / 100
            )
            for j in range(len(selection)):
                y_pred.append(posting_ids[selection[j]].tolist())
            torch.cuda.empty_cache()
        y_pred_df = pd.DataFrame()
        y_pred_df["y_pred"] = y_pred
        return y_pred_df

    @staticmethod
    def get_concat_voting_inference_y_pred_df(
        batch_idxs: List[Tuple[int, int]],
        features,
        thresholds: List[List[int]],
        posting_ids: np.ndarray,
        concat_configs: List[InferenceConcatConfig],
        tqdm_disable: bool = False,
        valid_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Tuple, Tuple]:
        best_y_pred_df = pd.DataFrame()
        best_score = 0
        best_thresholds = ()
        best_weights = ()
        weights_list = [
            [[50]]
            if len(concat_config.weights) == 1
            else [
                [20, 50],
                [30, 50],
                [40, 50],
                [50, 50],
                [50, 40],
                [50, 30],
                [50, 20],
            ]
            for concat_config in concat_configs
        ]
        for weights in itertools.product(*weights_list):
            __features = []
            print("weights", weights)
            for i, concat_config in enumerate(concat_configs):
                _features = copy.deepcopy(features)
                for idx, weight in zip(concat_config.idxs, weights[i]):
                    _features[idx] = _features[idx] * weight / np.sum(weights[i])
                _features = np.concatenate(
                    [_features[idx] for idx in concat_config.idxs], axis=1
                )
                _features = normalize(_features)
                __features.append(_features)
            for th in itertools.product(*thresholds):
                _y_pred = [[] for _ in __features]
                for (left, right) in tqdm(batch_idxs, disable=tqdm_disable):
                    for i in range(len(__features)):
                        selection = cupy.asnumpy(
                            cupy.matmul(
                                cupy.array(__features[i][left:right, :]),
                                cupy.array(__features[i].T),
                            )
                            > th[i] / 100
                        )
                        for j in range(len(selection)):
                            _y_pred[i].append(posting_ids[selection[j]].tolist())
                        torch.cuda.empty_cache()
                _y_pred_df = pd.DataFrame()
                for i in range(len(_y_pred)):
                    _y_pred_df[f"y_pred_{i}"] = _y_pred[i]
                if valid_df is None:
                    best_y_pred_df = _y_pred_df
                else:
                    _y_pred_df["y_pred"] = _y_pred_df.filter(like="y_pred_").apply(
                        lambda row: np.unique(np.concatenate(row)).tolist(), axis=1
                    )
                    (
                        _y_pred_df,
                        f1_score,
                        precision,
                        recall,
                    ) = InferenceFactory._calc_metric(valid_df, _y_pred_df)
                    print(
                        f"----------- valid f1: {f1_score} best: {best_score} precision: {precision} recall: {recall} threshold: {th} weights: {weights} ------------"
                    )
                    if best_score < f1_score:
                        best_score = f1_score
                        best_y_pred_df = _y_pred_df.copy()
                        best_thresholds = th
                        best_weights = weights
        return best_y_pred_df, best_thresholds, best_weights

    @staticmethod
    def get_avg_voting_y_pred_df(
        batch_idxs: List[Tuple[int, int]],
        features,
        thresholds: List[List[int]],
        posting_ids: np.ndarray,
        concat_configs: List[InferenceConcatConfig],
        inference_config: InferenceConfig,
        tqdm_disable: bool = False,
        valid_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Tuple, Tuple]:
        best_y_pred_df = pd.DataFrame()
        best_score = 0
        best_thresholds = ()
        best_weights = ()
        weights_list = [
            [[50]]
            if len(concat_config.weights) == 1
            else [
                [20, 50],
                [30, 50],
                [40, 50],
                [50, 50],
                [50, 40],
                [50, 30],
                [50, 20],
            ]
            if len(concat_config.weights) == 2
            else [
                [50, 50, 50],
                [40, 50, 50],
                [50, 40, 50],
                [50, 50, 40],
                [40, 40, 50],
                [40, 50, 40],
                [50, 40, 40],
            ]
            for concat_config in concat_configs
        ]
        for weights in itertools.product(*weights_list):
            print("weights", weights)
            for th in itertools.product(*thresholds):
                _y_pred = [[] for _ in concat_configs]
                for i, concat_config in enumerate(concat_configs):
                    for (left, right) in tqdm(batch_idxs, disable=tqdm_disable):
                        dot_product = np.zeros(
                            (right - left, len(posting_ids)), dtype=np.float64
                        )
                        for idx, weight in zip(concat_config.idxs, weights[i]):
                            dot_product += (
                                weight
                                * cupy.asnumpy(
                                    cupy.matmul(
                                        cupy.array(features[idx][left:right, :]),
                                        cupy.array(features[idx].T),
                                    )
                                )
                                / np.sum(weights[i])
                            )
                        selection = dot_product > th[i] / 100
                        for j in range(len(selection)):
                            IDX = selection[j]
                            if np.sum(IDX) < concat_config.min_indices:
                                IDX = np.argsort(dot_product[j])[
                                    -concat_config.min_indices :
                                ]
                            _y_pred[i].append(posting_ids[IDX].tolist())
                        torch.cuda.empty_cache()
                _y_pred_df = pd.DataFrame()
                for i in range(len(_y_pred)):
                    _y_pred_df[f"y_pred_{i}"] = _y_pred[i]
                if valid_df is None:
                    best_y_pred_df = _y_pred_df
                else:

                    def combine_predictions_major(row):
                        x = np.concatenate(row.values.reshape(-1))
                        x, counts = np.unique(x, return_counts=True)
                        ret_idx = counts >= inference_config.min_voting_count
                        return x[ret_idx]

                    _y_pred_df["y_pred"] = _y_pred_df.apply(
                        combine_predictions_major, axis=1
                    )
                    _y_pred_df["y_pred"] = _y_pred_df["y_pred"].map(
                        lambda y_pred: np.unique(y_pred).tolist()
                    )
                    (
                        _y_pred_df,
                        f1_score,
                        precision,
                        recall,
                    ) = InferenceFactory._calc_metric(valid_df, _y_pred_df)
                    print(
                        f"----------- valid f1: {f1_score} best: {best_score} precision: {precision} recall: {recall} threshold: {th} weights: {weights} ------------"
                    )
                    if best_score < f1_score:
                        best_score = f1_score
                        best_y_pred_df = _y_pred_df.copy()
                        best_thresholds = th
                        best_weights = weights
        return best_y_pred_df, best_thresholds, best_weights
