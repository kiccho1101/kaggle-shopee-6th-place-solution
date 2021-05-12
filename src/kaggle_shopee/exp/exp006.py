# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.file_util import FileUtil
from kaggle_shopee.utils.global_util import GlobalUtil
from kaggle_shopee.utils.mlflow_util import MlflowUtil

args = ArgsUtil.get_args(EnvEnum.LOCAL, "sexp001", [])
print(args)

config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, False)
print(config.dir_config)
GlobalUtil.seed_everything(config.seed)
data = DataFactory.load_data(config)
data, config = Pp.main(data, config)

MlflowUtil.start_run(config.mlflow_config, config.exp, config.name, "sexp", True)
MlflowUtil.log_params_config(config)
for fold in range(config.cv_config.n_splits):
    if fold not in args.folds:
        continue
    print(f"======================= fold {fold} =======================")
    train_dataloader, valid_dataloader = DataLoaderFactory.get_cv_dataloaders(
        data, fold, config
    )
    checkpoint_path = (
        f"{args.exp}_{fold}_{config.model_config.model_arch}"
        + "_{epoch:02d}_{threshold}_{valid_f1:.4f}"
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=str(config.dir_config.checkpoint_dir),
        filename=checkpoint_path,
        save_top_k=1,
        monitor="valid_f1",
        mode="max",
        verbose=True,
    )
    model = lit_models.ShopeeLitModel(data, config, fold)
    resume_from_checkpoint = FileUtil.get_resume_from_checkpoint(args.env, config, fold)
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=config.train_config.epochs,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=resume_from_checkpoint,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=1,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
MlflowUtil.end_run()
