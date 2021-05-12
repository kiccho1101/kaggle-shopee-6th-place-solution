import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
from kaggle_shopee.factories.config_factory import ConfigFactory
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.global_util import GlobalUtil
from kaggle_shopee.utils.metric_util import MetricUtil
from kaggle_shopee.utils.mlflow_util import MlflowUtil
from kaggle_shopee.utils.test_util import TestUtil

args = ArgsUtil.get_args()
print(args)

assert args.exp.startswith("eexp"), "{} does not start with eexp".format(args.exp)
config = ConfigFactory.get_config_from_yaml_file(args.exp, env=args.env, verbose=False)
print(config.dir_config)
GlobalUtil.seed_everything(config.seed)

exps = [m.split("_")[0] for m in config.inference_config.model_checkpoints]
score_means = []
MlflowUtil.start_run(config.mlflow_config, config.exp, config.name, True)
MlflowUtil.log_params_e_config(config)
for fold in range(5):
    if fold not in args.folds:
        continue

    y_preds: List[List[List[str]]] = []
    dfs: List[pd.DataFrame] = [
        pd.read_csv(config.dir_config.output_dir.parent / exp / f"valid_df_{fold}.csv")
        for exp in exps
    ]
    for df in dfs:
        TestUtil.assert_any(len(dfs[0]), len(df))

    y_true = dfs[0]["target_list"] = (
        dfs[0]["target"]
        .map(
            lambda s: s.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace("\n", "")
            .split()
        )
        .tolist()
    )

    for df in dfs:
        y_preds.append(
            df["y_pred"]
            .map(
                lambda s: s.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(", ")
            )
            .tolist()
        )

    y_pred = []
    for i in range(len(y_preds[0])):
        y_pred.append(
            np.unique([v for j in range(len(y_preds)) for v in y_preds[j][i]]).tolist()
        )

    scores = MetricUtil.f1_scores(y_true, y_pred)
    score = np.mean(scores)
    print(f"==================== fold: {fold} ======================")
    print(f"==================== valid_f1: {score} ======================")
    MlflowUtil.log_metric(f"v_best_{fold}", score)
    MlflowUtil.log_best_score_mean(config)
MlflowUtil.end_run()
