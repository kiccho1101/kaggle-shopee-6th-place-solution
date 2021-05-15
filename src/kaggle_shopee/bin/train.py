import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import ConfigFactory
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.feature_engineering import Fe
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.file_util import FileUtil
from kaggle_shopee.utils.global_util import GlobalUtil
from kaggle_shopee.utils.string_util import StringUtil

args = ArgsUtil.get_args()
print(args)

config = ConfigFactory.get_config_from_yaml_file(args.exp, env=args.env)
print(config.dir_config)
GlobalUtil.seed_everything(config.seed)
data = DataFactory.load_data(config)
if len(data.test) != 3:
    config.is_submitting = True
data, config = Pp.main(data, config)

fold = -1
train_dataloader, valid_dataloader = DataLoaderFactory.get_train_dataloaders(
    data, config
)
checkpoint_path = f"{args.exp}_-1" + "_{epoch:02d}"
checkpoint_callback = callbacks.ModelCheckpoint(
    dirpath=str(config.dir_config.checkpoint_out_dir),
    filename=checkpoint_path,
    save_top_k=-1,
    verbose=True,
)
model = lit_models.ShopeeLitModel(data, config, fold=0, with_mlflow=False)
resume_from_checkpoint = FileUtil.get_resume_from_checkpoint(args.env, config, -1)
trainer = pl.Trainer(
    gpus=-1,
    max_epochs=config.train_config.epochs,
    checkpoint_callback=checkpoint_callback,
    resume_from_checkpoint=resume_from_checkpoint,
    num_sanity_val_steps=0,
    progress_bar_refresh_rate=1,
)
trainer.fit(model, train_dataloader)

last_epoch = config.train_config.epochs - 1
last_epoch_str = str(last_epoch).zfill(2)
checkpoint_path = f"{args.exp}_-1_epoch{last_epoch_str}.ckpt"
print("checkpoint_path", checkpoint_path)
trainer.save_checkpoint(checkpoint_path)
