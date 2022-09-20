import sys
from pathlib import Path

try:
    sys.path.append(str(Path(__file__).parents[2]))
except:
    sys.path.append("/content/kaggle-shopee/src")

import gc

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch.cuda
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.file_util import FileUtil
from kaggle_shopee.utils.global_util import GlobalUtil
from kaggle_shopee.utils.mlflow_util import MlflowUtil
from kaggle_shopee.utils.string_util import StringUtil

args = ArgsUtil.get_args()
# args = ArgsUtil.get_args(EnvEnum.COLAB, "exp000", [0])
print(args)

config = ConfigFactory.get_config_from_yaml_file(args.exp, env=args.env)
print(config.dir_config)
GlobalUtil.seed_everything(config.seed)
data = DataFactory.load_data(config)
data, config = Pp.main(data, config)
# data, config = Fe.main(data, config)


for fold in range(config.cv_config.n_splits):
    if fold not in args.folds:
        continue
    print(f"======================= fold {fold} =======================")
    train_dataloader, valid_dataloader = DataLoaderFactory.get_cv_dataloaders(
        data, fold, config
    )
    checkpoint_path = f"{args.exp}_{fold}" + "_{epoch:02d}_{threshold}_{valid_f1:.4f}"
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=str(config.dir_config.checkpoint_out_dir),
        filename=checkpoint_path,
        save_top_k=1,
        monitor="valid_f1",
        mode="max",
        verbose=True,
    )
    model = lit_models.ShopeeLitModel(
        data=data, config=config, fold=fold, with_mlflow=False
    )
    # resume_from_checkpoint = FileUtil.get_resume_from_checkpoint(args.env, config, fold)
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=config.train_config.epochs,
        # resume_from_checkpoint=resume_from_checkpoint,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
MlflowUtil.end_run()
