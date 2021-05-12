from typing import Any, Dict, Optional

import numpy as np
from kaggle_shopee.factories.config_factory import Config, MlflowConfig

try:
    import mlflow
except Exception as e:
    print(e)


class MlflowUtil:
    @staticmethod
    def start_run(
        mlflow_config: MlflowConfig,
        exp: str,
        name: str,
        verbose: bool = True,
    ):
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(mlflow_config.uri)
        exp_name = mlflow_config.exp_name
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name)
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id

        search_df: Any = mlflow.search_runs(
            [experiment_id], filter_string=f"tags.exp = '{exp}'"
        )
        resume = len(search_df) > 0
        if resume:
            run_id = search_df["run_id"].iloc[0]
            mlflow.start_run(run_id=run_id, experiment_id=experiment_id)
        else:
            mlflow.start_run(run_name=f"{exp}: {name}", experiment_id=experiment_id)
            mlflow.set_tag("exp", exp)

        if verbose:
            print(f"mlflow started. experiment name: {exp_name}")
            print(f"{mlflow_config.uri}/#/experiments/{experiment_id}")

    @staticmethod
    def end_run():
        mlflow.end_run()

    @staticmethod
    def log_params_if_not_exists(
        params: Dict[str, Any], current_params: Dict[str, float]
    ):
        for key, value in params.items():
            if key not in current_params:
                mlflow.log_param(key, value)

    @staticmethod
    def log_params_config(config: Config):
        run = mlflow.active_run()
        current_params: Dict[str, float] = run.data.params
        MlflowUtil.log_params_if_not_exists(
            {
                "exp": config.exp,
                "seed": config.seed,
                "model_name": config.model_config.model_name,
                "model_arch": config.model_config.model_arch,
                "bert_model_arch": config.model_config.bert_model_arch,
                "channel_size": config.model_config.channel_size,
                "dropout": config.model_config.dropout,
                "pretrained": config.model_config.pretrained,
                "n_splits": config.cv_config.n_splits,
                "train_batch_size": config.train_config.train_batch_size,
                "valid_batch_size": config.train_config.valid_batch_size,
                "epochs": config.train_config.epochs,
                "num_workers": config.train_config.num_workers,
                "stage1_checkpoint": config.model_config.stage1_checkpoint,
                "optimizer.name": config.optimizer_config.name,
                "scheduler.name": config.scheduler_config.name,
                "met.name": config.met_config.name,
                "loss.name": config.loss_config.name,
                "pooling.name": config.pooling_config.name,
                "bert_pooling.name": config.bert_pooling_config.name,
                "ensemble_method": config.inference_config.ensemble_method,
                "img_size": config.dataset_config.img_size,
            },
            current_params,
        )
        MlflowUtil.log_params_if_not_exists(
            {f"optimizer.{k}": v for k, v in config.optimizer_config.params.items()},
            current_params,
        )
        MlflowUtil.log_params_if_not_exists(
            {f"scheduler.{k}": v for k, v in config.scheduler_config.params.items()},
            current_params,
        )
        MlflowUtil.log_params_if_not_exists(
            {f"met.{k}": v for k, v in config.met_config.params.items()},
            current_params,
        )
        MlflowUtil.log_params_if_not_exists(
            {f"loss.{k}": v for k, v in config.loss_config.params.items()},
            current_params,
        )

    @staticmethod
    def log_params_e_config(config: Config):
        run = mlflow.active_run()
        current_params: Dict[str, float] = run.data.params
        MlflowUtil.log_params_if_not_exists(
            {
                "exp": config.exp,
                "models": str(config.inference_config.model_checkpoints),
            },
            current_params,
        )

    @staticmethod
    def log_metric(key: str, value: Any, step: Optional[int] = None):
        mlflow.log_metric(key, value, step)

    @staticmethod
    def get_best_score(config: Config, fold: int):
        MlflowUtil.start_run(config.mlflow_config, config.exp, config.name, False)
        run = mlflow.active_run()
        metrics: Dict[str, float] = run.data.metrics
        key = f"v_best_{fold}"
        if key in metrics:
            return metrics[key]
        return -1e10

    @staticmethod
    def log_best_score_mean(config: Config):
        MlflowUtil.start_run(config.mlflow_config, config.exp, config.name, False)
        run = mlflow.active_run()
        metrics: Dict[str, float] = run.data.metrics
        best_scores = []
        for fold in range(5):
            key = f"v_best_{fold}"
            if key in metrics:
                best_scores.append(metrics[key])
        if len(best_scores) > 0:
            MlflowUtil.log_metric("v_best_mean", np.mean(best_scores))
