# %%
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from typing import List

import numpy as np
import torch
from kaggle_shopee.factories import lit_models
from kaggle_shopee.factories.config_factory import ConfigFactory, EnvEnum
from kaggle_shopee.factories.data_factory import Data, DataFactory
from kaggle_shopee.factories.dataloader_factory import DataLoaderFactory
from kaggle_shopee.factories.preprocessing import Pp
from kaggle_shopee.utils.args_util import ArgsUtil
from kaggle_shopee.utils.string_util import StringUtil
from tqdm import tqdm

args = ArgsUtil.get_args(EnvEnum.LOCAL, "exp017", [])
config = ConfigFactory.get_config_from_yaml_file(args.exp, args.env, False)
data = DataFactory.load_data(config)
data, config = Pp.main(data, config)
test_dataloader = DataLoaderFactory.get_test_dataloader(data, config)


def load_model(model_checkpoint: str):
    if args.env == EnvEnum.KAGGLE:
        model_checkpoint = model_checkpoint.replace("=", "")
    _exp = model_checkpoint.split("_")[0]
    _config = ConfigFactory.get_config_from_yaml_file(_exp, args.env, False)
    _config.model_config.pretrained = False
    lit_model: lit_models.ShopeeLitModel = (
        lit_models.ShopeeLitModel.load_from_checkpoint(
            str(config.dir_config.checkpoint_dir / model_checkpoint),
            data=data,
            config=_config,
            fold=-1,
            with_mlflow=False,
            bert_path=str(config.dir_config.input_dir / "kaggle-shopee-dataset"),
        )
    )
    return lit_model.model


device = "cuda"
img_model = load_model(config.inference_config.model_checkpoints[1])
img_model.to(device)
img_model.eval()

txt_model = load_model(config.inference_config.model_checkpoints[0])
txt_model.to(device)
txt_model.eval()

img_features = []
txt_features = []
for batch in tqdm(test_dataloader):
    img = batch["img"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    img_features += [img_model(img).detach().cpu()]
    txt_features += [txt_model(input_ids, attention_mask).detach().cpu()]

img_features = torch.cat(img_features).cpu().numpy()
txt_features = torch.cat(txt_features).cpu().numpy()

matches: List[str] = []
sim_threshold = config.inference_config.threshold / 100
posting_ids = data.test["posting_id"].values
n_batches = 20
n_rows = img_features.shape[0]
bs = n_rows // n_batches
batch_idxs = []
for i in range(n_batches):
    left = bs * i
    right = bs * (i + 1)
    if i == n_batches - 1:
        right = n_rows
    batch_idxs.append((left, right))
for (left, right) in batch_idxs:
    img_selection = img_features[left:right, :] @ img_features.T > sim_threshold
    for i in range(len(img_selection)):
        matches.append(" ".join(np.unique([*posting_ids[img_selection[i]]])))

data.sample_submission["matches"] = matches
data.sample_submission
