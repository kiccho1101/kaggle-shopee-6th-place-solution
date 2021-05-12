import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import torch
import torch.cuda
from kaggle_shopee.factories.config_factory import Config, ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.inference_factory import InferenceFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.file_util import FileUtil
from kaggle_shopee.utils.global_util import GlobalUtil
from kaggle_shopee.utils.mlflow_util import MlflowUtil

args = ArgsUtil.get_args()
print(args)

config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, True)
print(config.inference_config)
GlobalUtil.seed_everything(config.seed)
data = DataFactory.load_data(config)
data, config = Pp.main(data, config)


MlflowUtil.start_run(config.mlflow_config, config.exp, config.name, True)
MlflowUtil.log_params_config(config)
for fold in range(config.cv_config.n_splits):
    if fold not in args.folds:
        continue
    print(f"======================= fold {fold} =======================")
    features = []
    valid_df = data.train[data.train[config.cv_config.fold_col] == fold].copy()
    posting_ids = valid_df["posting_id"].values
    for i in range(len(config.inference_config.epoch_configs)):
        for j in range(len(config.inference_config.epoch_configs[i].model_checkpoints)):
            _exp = (
                config.inference_config.epoch_configs[i]
                .model_checkpoints[j]
                .split("_")[0]
            )
            InferenceFactory.prepare_bert(args.env, _exp)
            _config = ConfigFactory.get_config_from_yaml_file(_exp, args.env, False)
            checkpoint_path = FileUtil.get_best_cv_checkpoint(args.env, _exp, fold)
            assert (
                checkpoint_path is not None
            ), f"exp: {_exp} fold: {fold} checkpoint doesn't exist"
            config.inference_config.epoch_configs[i].model_checkpoints[
                j
            ] = checkpoint_path
    for epoch_config in config.inference_config.epoch_configs:
        _config = ConfigFactory.get_config_from_yaml_file(
            epoch_config.dataloader_exp, args.env, False
        )
        _, valid_dataloader = DataLoaderFactory.get_cv_dataloaders(data, fold, _config)
        features += InferenceFactory.epoch(
            args.env, epoch_config, valid_dataloader, data
        )
    for i in range(len(features)):
        features[i] = torch.cat(features[i]).cpu().numpy()
    batch_idxs = InferenceFactory.get_batch_idxs(len(valid_df), n_batches=20)
    _valid_df = valid_df.reset_index(drop=True).copy()
    if config.inference_config.ensemble_method == "concat":
        weights_list = [
            [wg + add for add in np.arange(-20, 20, 10)]
            for wg in config.inference_config.weights
        ]
        y_pred_df, best_threshold, best_weights = InferenceFactory.get_concat_y_pred_df(
            batch_idxs,
            features,
            np.arange(30, 80, 5).tolist(),
            posting_ids,
            weights_list,
            tqdm_disable=True,
            valid_df=valid_df,
        )
    elif config.inference_config.ensemble_method == "concat_voting":
        _thresholds = [
            [th + add for add in np.arange(-10, 10, 5)]
            for th in config.inference_config.thresholds
        ]
        (
            y_pred_df,
            best_thresholds,
            best_weights,
        ) = InferenceFactory.get_concat_voting_inference_y_pred_df(
            batch_idxs,
            features,
            _thresholds,
            posting_ids,
            config.inference_config.concat_configs,
            tqdm_disable=True,
            valid_df=valid_df,
        )
        for i in range(len(config.inference_config.concat_configs)):
            _valid_df[f"y_pred_{i}"] = y_pred_df[f"y_pred_{i}"]
    elif config.inference_config.ensemble_method == "avg_voting":
        _thresholds = [
            [th + add for add in np.arange(-5, 10, 5)]
            for th in config.inference_config.thresholds
        ]
        (
            y_pred_df,
            best_thresholds,
            best_weights,
        ) = InferenceFactory.get_avg_voting_y_pred_df(
            batch_idxs,
            features,
            _thresholds,
            posting_ids,
            config.inference_config.concat_configs,
            config.inference_config,
            tqdm_disable=True,
            valid_df=valid_df,
        )
        for i in range(len(config.inference_config.concat_configs)):
            _valid_df[f"y_pred_{i}"] = y_pred_df[f"y_pred_{i}"]
    else:
        _thresholds = [
            [th + add for add in np.arange(-10, 10, 5)]
            for th in config.inference_config.thresholds
        ]
        y_pred_df, best_thresholds = InferenceFactory.get_y_pred_df(
            batch_idxs,
            features,
            _thresholds,
            posting_ids,
            tqdm_disable=True,
            valid_df=valid_df,
        )
        for i in range(len(features)):
            _valid_df[f"y_pred_{i}"] = y_pred_df[f"y_pred_{i}"]
    best_score = y_pred_df["f1_score"].mean()
    best_precision = y_pred_df["precision"].mean()
    best_recall = y_pred_df["recall"].mean()
    _valid_df["y_pred"] = y_pred_df["y_pred"]
    _valid_df["f1_score"] = y_pred_df["f1_score"]
    _valid_df["precision"] = y_pred_df["precision"]
    _valid_df["recall"] = y_pred_df["recall"]
    _valid_df.to_csv(
        config.dir_config.output_dir / f"valid_df_{fold}.csv",
        index=False,
    )
    MlflowUtil.log_metric(f"v_best_{fold}", y_pred_df["f1_score"].mean())
    if config.inference_config.ensemble_method == "concat":
        print(
            f"----------- best valid f1: {best_score} precision: {best_precision} recall: {best_recall} weights: {best_weights} threshold: {best_threshold} ------------"
        )
        MlflowUtil.log_metric(f"v_th_{fold}", float(best_threshold))
        for i in range(len(best_weights)):
            MlflowUtil.log_metric(f"v_wg_{fold}_{i}", float(best_weights[i]))
    elif config.inference_config.ensemble_method == "concat_voting":
        print(
            f"----------- best valid f1: {best_score} precision: {best_precision} recall: {best_recall} weights: {best_weights} threshold: {best_thresholds} ------------"
        )
        for j in range(len(best_thresholds)):
            MlflowUtil.log_metric(f"v_th_{fold}_{j}", float(best_thresholds[j]))
        for j in range(len(best_weights)):
            for k in range(len(best_weights[j])):
                MlflowUtil.log_metric(f"v_wg_{fold}_{j}_{k}", float(best_weights[j][k]))
    else:
        print(
            f"----------- best valid f1: {best_score} precision: {best_precision} recall: {best_recall} threshold: {best_thresholds} ------------"
        )
        for j in range(len(best_thresholds)):
            MlflowUtil.log_metric(f"v_th_{fold}_{j}", float(best_thresholds[j]))
    MlflowUtil.log_best_score_mean(config)
MlflowUtil.end_run()
