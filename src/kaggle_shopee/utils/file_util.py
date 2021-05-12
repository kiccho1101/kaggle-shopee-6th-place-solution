import pickle
import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from kaggle_shopee.factories.config_factory import Config, ConfigFactory, EnvEnum
from kaggle_shopee.utils.string_util import StringUtil
from kaggle_shopee.utils.time_util import TimeUtil


class FileUtil:
    @staticmethod
    def read_csv(filepath: Union[str, Path], verbose: bool = True):
        if verbose:
            with TimeUtil.timer(f"Read {str(filepath)}"):
                return pd.read_csv(filepath)
        return pd.read_csv(filepath)

    @staticmethod
    def save_npy(arr: np.ndarray, filepath: Union[str, Path], verbose: bool = True):
        with open(filepath, "wb") as f:
            if verbose:
                with TimeUtil.timer(f"Save {str(filepath)}"):
                    np.save(f, arr)
            else:
                np.save(f, arr)

    @staticmethod
    def load_npy(filepath: Union[str, Path], verbose: bool = True) -> np.ndarray:
        with open(filepath, "rb") as f:
            if verbose:
                with TimeUtil.timer(f"Load {str(filepath)}"):
                    arr = np.load(f)
            else:
                arr = np.load(f)
        return arr

    @staticmethod
    def save_pickle(obj: Any, filepath: Union[str, Path], verbose: bool = True):
        with open(filepath, "wb") as f:
            if verbose:
                with TimeUtil.timer(f"Save {str(filepath)}"):
                    pickle.dump(obj, f)
            else:
                pickle.dump(obj, f)

    @staticmethod
    def load_pickle(filepath: Union[str, Path], verbose: bool = True) -> Any:
        with open(filepath, "rb") as f:
            if verbose:
                with TimeUtil.timer(f"Load {str(filepath)}"):
                    obj = pickle.load(f)
            else:
                obj = pickle.load(f)
        return obj

    @staticmethod
    def get_resume_from_checkpoint(
        env: EnvEnum, config: Config, fold: int
    ) -> Optional[str]:
        if config.train_config.resume_from_exp is None:
            return None
        checkpoints = [
            str(path)
            for path in Path(config.dir_config.checkpoint_dir).glob(
                f"{config.train_config.resume_from_exp}_{fold}_*"
            )
        ]
        if len(checkpoints) == 0:
            print(f"{config.train_config.resume_from_exp}_{fold}_*")
            return None
        max_acc_idx = np.argmax(
            [
                re.findall(r"epoch=(\d{1,2})", checkpoint)[0]
                for checkpoint in checkpoints
            ]
        )
        checkpoint = checkpoints[max_acc_idx]
        return checkpoint

    @staticmethod
    def get_best_cv_checkpoint(env: EnvEnum, exp: str, fold: int) -> Optional[str]:
        config = ConfigFactory.get_config_from_yaml_file(exp, env, False)
        checkpoints = [
            str(path).replace(str(config.dir_config.checkpoint_dir) + "/", "")
            for path in Path(config.dir_config.checkpoint_dir).glob(f"{exp}_{fold}_*")
        ]
        if len(checkpoints) == 0:
            print(f"{exp}_{fold}_*")
            return None
        max_acc_idx = np.argmax(
            [
                re.findall(
                    r"epoch(\d{1,2})" if env == EnvEnum.KAGGLE else r"epoch=(\d{1,2})",
                    checkpoint,
                )[0]
                for checkpoint in checkpoints
            ]
        )
        checkpoint = checkpoints[max_acc_idx]
        return checkpoint
