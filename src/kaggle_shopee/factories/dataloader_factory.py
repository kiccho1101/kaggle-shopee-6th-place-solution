from typing import Optional, Tuple

from kaggle_shopee.factories import datasets
from kaggle_shopee.factories.config_factory import Config
from kaggle_shopee.factories.data_factory import Data
from kaggle_shopee.factories.transform_factory import TransformFactory
from torch.utils.data import DataLoader


class DataLoaderFactory:
    @staticmethod
    def get_cv_dataloaders(
        data: Data, fold: int, config: Config, target: str = "label_group_le"
    ) -> Tuple[DataLoader, DataLoader]:
        _train_df = data.train[data.train[config.cv_config.fold_col] != fold].copy()
        _valid_df = data.train[data.train[config.cv_config.fold_col] == fold].copy()

        if len(_train_df) % config.train_config.train_batch_size == 1:
            _train_df = _train_df.sample(len(_train_df) - 1).reset_index(drop=True)

        train_dataset = datasets.ShopeeDataset(
            df=_train_df,
            output_label=True,
            dataset_config=config.dataset_config,
            model_config=config.model_config,
            train_config=config.train_config,
            dir_config=config.dir_config,
            transforms=TransformFactory.get_transforms(config.transform_config.train),
            fold=fold,
        )
        valid_dataset = datasets.ShopeeDataset(
            df=_valid_df,
            output_label=True,
            dataset_config=config.dataset_config,
            model_config=config.model_config,
            train_config=config.train_config,
            dir_config=config.dir_config,
            transforms=TransformFactory.get_transforms(config.transform_config.valid),
        )

        print("num_workers:", config.train_config.num_workers)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_config.train_batch_size,
            num_workers=config.train_config.num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config.train_config.valid_batch_size,
            num_workers=config.train_config.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        return train_dataloader, valid_dataloader

    @staticmethod
    def get_train_dataloaders(
        data: Data, config: Config
    ) -> Tuple[DataLoader, DataLoader]:
        _train_df = data.train.copy()
        _valid_df = data.train[data.train[config.cv_config.fold_col] == 0].copy()

        train_dataset = datasets.ShopeeDataset(
            df=_train_df,
            output_label=True,
            dataset_config=config.dataset_config,
            model_config=config.model_config,
            train_config=config.train_config,
            dir_config=config.dir_config,
            transforms=TransformFactory.get_transforms(config.transform_config.train),
            fold=-1,
        )
        valid_dataset = datasets.ShopeeDataset(
            df=_valid_df,
            output_label=True,
            dataset_config=config.dataset_config,
            model_config=config.model_config,
            train_config=config.train_config,
            dir_config=config.dir_config,
            transforms=TransformFactory.get_transforms(config.transform_config.valid),
        )

        print("num_workers:", config.train_config.num_workers)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_config.train_batch_size,
            num_workers=config.train_config.num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config.train_config.valid_batch_size,
            num_workers=config.train_config.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        return train_dataloader, valid_dataloader

    @staticmethod
    def get_test_dataloader(
        data: Data,
        config: Config,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        for c in config.transform_config.inference:
            if c.name == "Resize":
                print(config.exp, c.name, c.args)
        test_dataset = datasets.ShopeeDataset(
            df=data.test,
            output_label=False,
            dataset_config=config.dataset_config,
            model_config=config.model_config,
            train_config=config.train_config,
            dir_config=config.dir_config,
            transforms=TransformFactory.get_transforms(
                config.transform_config.inference
            ),
            tokenizer_path=str(
                config.dir_config.dataset_dir / config.model_config.bert_model_arch
            ),
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.train_config.valid_batch_size
            if batch_size is None
            else batch_size,
            num_workers=config.train_config.num_workers
            if num_workers is None
            else num_workers,
            shuffle=False,
            pin_memory=False,
        )
        return test_dataloader
