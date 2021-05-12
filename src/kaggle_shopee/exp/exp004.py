# %%
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
import argparse
from dataclasses import dataclass
from typing import List, Optional

from kaggle_shopee.factories.config_factory import EnvEnum


@dataclass
class Args:
    env: EnvEnum
    exp: str
    folds: List[int]


class ArgsUtil:
    @staticmethod
    def get_args(
        env: Optional[EnvEnum] = None,
        exp: Optional[str] = None,
        folds: Optional[List[int]] = None,
    ) -> Args:
        if env is not None and exp is not None and folds is not None:
            return Args(env=env, exp=exp, folds=folds)

        _env = EnvEnum.LOCAL
        _exp = "exp001"
        _folds = []

        parser = argparse.ArgumentParser()
        parser.add_argument("--env", default=None)
        parser.add_argument("--exp", default=None)
        parser.add_argument("--folds", default=None)
        args = parser.parse_args()
        if args.env is not None:
            _env = EnvEnum.from_str(args.env)
        if args.exp is not None:
            _exp = args.exp
        if args.folds is not None:
            _folds = [int(f) for f in args.folds.replace(" ", "").split(",")]
        return Args(env=_env, exp=_exp, folds=_folds)


args = ArgsUtil.get_args()
print(args)
