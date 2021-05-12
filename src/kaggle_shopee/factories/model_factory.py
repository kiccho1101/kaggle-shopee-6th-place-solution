from typing import Optional

import pandas as pd
from kaggle_shopee.factories.config_factory import (BertPoolingConfig,
                                                    MetricLearningConfig,
                                                    ModelConfig, PoolingConfig)
from kaggle_shopee.models import img_models, img_txt_models, txt_models


class ModelFactory:
    @staticmethod
    def get_model(
        out_features: int,
        model_config: ModelConfig,
        met_config: MetricLearningConfig,
        pooling_config: PoolingConfig,
        bert_pooling_config: BertPoolingConfig,
        bert_path: Optional[str] = None,
        train_df: pd.DataFrame = pd.DataFrame(),
    ):
        if model_config.model_name == "ShopeeImgNet":
            return img_models.ShopeeImgNet(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                pooling_config=pooling_config,
            )
        elif model_config.model_name == "ShopeeImgNet2":
            return img_models.ShopeeImgNet2(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                pooling_config=pooling_config,
                train_df=train_df,
            )
        elif model_config.model_name == "ShopeeImgNet3":
            return img_models.ShopeeImgNet3(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                pooling_config=pooling_config,
            )
        elif model_config.model_name == "ShopeeImgNet4":
            return img_models.ShopeeImgNet4(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                pooling_config=pooling_config,
            )
        elif model_config.model_name == "ShopeeImgNet5":
            return img_models.ShopeeImgNet5(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                pooling_config=pooling_config,
            )
        elif model_config.model_name == "ShopeeTextNet":
            return txt_models.ShopeeTextNet(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                bert_path=bert_path,
                bert_pooling_config=bert_pooling_config,
                train_df=train_df,
            )
        elif model_config.model_name == "ShopeeImgTextNet":
            return img_txt_models.ShopeeImgTextNet(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                pooling_config=pooling_config,
                bert_pooling_config=bert_pooling_config,
                bert_path=bert_path,
            )
        elif model_config.model_name == "ShopeeKurupicalExp044Net":
            return img_txt_models.ShopeeKurupicalExp044Net(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                bert_path=bert_path,
            )
        elif model_config.model_name == "ShopeeVitImgTextNet":
            return img_txt_models.ShopeeVitImgTextNet(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                bert_pooling_config=bert_pooling_config,
                bert_path=bert_path,
            )
        elif model_config.model_name == "ShopeeImgTextNet5":
            return img_txt_models.ShopeeImgTextNet5(
                out_features=out_features,
                model_config=model_config,
                met_config=met_config,
                pooling_config=pooling_config,
                bert_pooling_config=bert_pooling_config,
                bert_path=bert_path,
            )
        else:
            raise ValueError(
                "{} is an invalid model name".format(model_config.model_name)
            )
