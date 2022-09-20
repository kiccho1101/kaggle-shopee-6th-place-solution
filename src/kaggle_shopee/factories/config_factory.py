import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class LogitTypeEnum(Enum):
    MARGIN = "margin"
    CLASS = "class"


class EnvEnum(Enum):
    LOCAL = "local"
    COLAB = "colab"
    KAGGLE = "kaggle"

    @staticmethod
    def from_str(s: str):
        if s.lower() == "local":
            return EnvEnum.LOCAL
        elif s.lower() == "colab":
            return EnvEnum.COLAB
        elif s.lower() == "kaggle":
            return EnvEnum.KAGGLE
        else:
            raise ValueError("Unknown enum value: %s" % s)


class ReducerEnum(Enum):
    NOTHING = "NOTHING"
    PCA = "PCA"
    UMAP = "UMAP"
    TSNE = "TSNE"
    TSVD = "TruncatedSVD"


@dataclass
class DirConfig:
    root_dir: Path
    input_dir: Path
    output_dir: Path
    data_dir: Path
    s_data_dir: Path
    checkpoint_dir: Path
    checkpoint_out_dir: Path
    dataset_dir: Path
    yamls_dir: Path
    train_images_dir: Path
    test_images_dir: Path
    additional_train_dir: Path
    train_fold_dir: Path


@dataclass
class CvConfig:
    name: str
    target: str
    n_splits: int
    fold_col: str = "fold"


@dataclass
class TrainConfig:
    train_batch_size: int
    valid_batch_size: int
    epochs: int
    num_workers: int
    resume_from_exp: Optional[str] = None
    target: str = "label_group_le"


@dataclass
class DatasetConfig:
    img_size: int
    text_col: str = "title"
    padding: bool = True
    truncation: bool = True
    max_length: int = 128
    mining_exp: str = ""
    mining_epoch: int = -1
    num_negatives: int = -1

    def is_with_triplets(self):
        return (
            self.mining_exp != ""
            and self.mining_epoch != -1
            and self.num_negatives != -1
        )


@dataclass
class TransformComponent:
    name: str
    args: Optional[Dict]


@dataclass
class TransformConfig:
    train: List[TransformComponent]
    valid: List[TransformComponent]
    inference: List[TransformComponent]


@dataclass
class ModelConfig:
    model_name: str
    model_arch: str
    pretrained: bool
    channel_size: int
    dropout: float
    forward_method: str = "normal"
    bert_model_arch: str = "cahya/distilbert-base-indonesian"
    bert_hidden_size: int = 768
    stage1_checkpoint: str = ""
    neck: str = "option-A"
    img_checkpoint: str = "exp000"
    txt_checkpoint: str = "exp000"
    normalize: bool = True
    transformer_n_head: int = 64
    transformer_dropout: float = 0.0
    transformer_num_layers: int = 1
    dropout_nlp: float = 0.2
    dropout_bert_stack: float = 0.2
    dropout_cnn_fc: float = 0.2
    dropout_global: float = 0.2
    dropout_local: float = 0.2
    n_conv: int = 0
    with_pretrain: bool = False
    w_concat: float = 1
    w_img: float = 0
    w_img_local: float = 0
    w_txt: float = 0
    w_triplet: float = 0


@dataclass
class LossConfig:
    name: str = "CrossEntropyLoss"
    params: Dict = field(default_factory=lambda: {})


@dataclass
class PoolingConfig:
    name: str = "AdaptiveAvgPool2d"
    params: Dict = field(default_factory=lambda: {"output_size": 1})


@dataclass
class BertPoolingConfig:
    name: str = "cls"
    params: Dict = field(default_factory=lambda: {})


@dataclass
class OptimizerConfig:
    name: str
    params: Dict


@dataclass
class SchedulerConfig:
    name: str
    params: Dict


@dataclass
class MetricLearningConfig:
    name: str
    params: Dict


@dataclass
class MlflowConfig:
    exp_name: str = "cv"
    uri: str = "{your mlflow uri here}"


@dataclass
class InferenceEpochConfig:
    dataloader_exp: str = "exp001"
    model_checkpoints: List[str] = field(default_factory=lambda: [])


@dataclass
class InferenceConcatConfig:
    weights: List[int] = field(default_factory=lambda: [])
    idxs: List[int] = field(default_factory=lambda: [])
    min_indices: int = 1


@dataclass
class InferenceConfig:
    threshold: int = 0
    threshold_search_method: str = "dot-product"  # dot-product or nearest-neighbors
    min_indices: int = 1
    model_checkpoints: List[str] = field(default_factory=lambda: [])
    thresholds: List[int] = field(default_factory=lambda: [])
    epoch_configs: List[InferenceEpochConfig] = field(default_factory=lambda: [])
    concat_configs: List[InferenceConcatConfig] = field(default_factory=lambda: [])
    with_tfidf: bool = False
    tfidf_threshold: int = 80
    tfidf_max_features: int = 10000
    ensemble_method: str = "voting"
    weights: List[int] = field(default_factory=lambda: [])
    min_voting_count: int = 1


@dataclass
class Config:
    name: str
    env: EnvEnum
    exp: str
    seed: int
    is_submitting: bool
    title_tfidf_n_components: int
    title_tfidf_reducer: ReducerEnum
    dir_config: DirConfig
    cv_config: CvConfig
    dataset_config: DatasetConfig
    transform_config: TransformConfig
    model_config: ModelConfig
    loss_config: LossConfig
    pooling_config: PoolingConfig
    bert_pooling_config: BertPoolingConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    train_config: TrainConfig
    met_config: MetricLearningConfig
    mlflow_config: MlflowConfig
    inference_config: InferenceConfig


class ConfigFactory:
    @staticmethod
    def _get_env() -> EnvEnum:
        if "google.colab" in sys.modules:
            return EnvEnum.COLAB
        elif "kaggle_web_client" in sys.modules:
            return EnvEnum.KAGGLE
        else:
            return EnvEnum.LOCAL

    @staticmethod
    def _get_root_dir(env: EnvEnum) -> Path:
        if env == EnvEnum.COLAB:
            return Path("/content/kaggle-shopee")
        elif env == EnvEnum.KAGGLE:
            return Path("/kaggle/input/kaggle-shopee")
        else:
            return Path(__file__).parents[3]

    @staticmethod
    def _get_output_dir(env: EnvEnum, exp: str, root_dir: Path) -> Path:
        if env == EnvEnum.COLAB:
            return Path("/content/drive/MyDrive/kaggle-shopee/output") / exp
        elif env == EnvEnum.KAGGLE:
            return Path("/kaggle/output") / exp
        else:
            return root_dir / "output" / exp

    @staticmethod
    def _get_checkpoint_dir(env: EnvEnum, root_dir: Path) -> Path:
        if env == EnvEnum.COLAB:
            return Path("/content/drive/MyDrive/kaggle-shopee/output/checkpoints")
        elif env == EnvEnum.KAGGLE:
            return Path("/kaggle/input/kaggle-shopee-dataset")
        else:
            return root_dir / "output" / "checkpoints"

    @staticmethod
    def _get_dataset_dir(env: EnvEnum, root_dir: Path) -> Path:
        if env == EnvEnum.COLAB:
            return Path("/content/kaggle-shopee-dataset")
        elif env == EnvEnum.KAGGLE:
            return Path("/kaggle/input/kaggle-shopee-dataset")
        else:
            return root_dir / "input" / "kaggle-shopee-dataset"

    @staticmethod
    def _get_s_data_dir(env: EnvEnum, input_dir: Path) -> Path:
        if env == EnvEnum.KAGGLE:
            return input_dir / "shopee-product-detection"
        else:
            return input_dir / "shopee-product-detection-student"

    @staticmethod
    def _get_input_dir(env: EnvEnum, root_dir: Path) -> Path:
        if env == EnvEnum.COLAB:
            return root_dir / "input"
        elif env == EnvEnum.KAGGLE:
            return Path("/kaggle/input")
        else:
            return root_dir / "input"

    @staticmethod
    def _get_additional_train_dir(env: EnvEnum, root_dir: Path) -> Path:
        if env == EnvEnum.COLAB:
            return Path("/content/drive/MyDrive/kaggle-shopee/input/shopee-product-matching/train.csv")
        else:
            return Path(".")

    @staticmethod
    def _get_train_fold_dir(env: EnvEnum, root_dir: Path) -> Path:
        if env == EnvEnum.COLAB:
            return Path("/content/drive/MyDrive/kaggle-shopee/input/shopee-product-matching/train.csv")
        elif env == EnvEnum.KAGGLE:
            return Path("/kaggle/input/shopee-product-matching/train.csv")
        else:
            return root_dir / "input" / "train_fold.csv"

    @staticmethod
    def _get_dir_config(env: EnvEnum, exp: str) -> DirConfig:
        root_dir = ConfigFactory._get_root_dir(env)
        output_dir = ConfigFactory._get_output_dir(env, exp, root_dir)
        checkpoint_dir = ConfigFactory._get_checkpoint_dir(env, root_dir)
        checkpoint_out_dir = Path("/tmp") if env == EnvEnum.KAGGLE else checkpoint_dir
        dataset_dir = ConfigFactory._get_dataset_dir(env, root_dir)
        input_dir = ConfigFactory._get_input_dir(env, root_dir)
        additional_train_dir = ConfigFactory._get_additional_train_dir(env, root_dir)
        train_fold_dir = ConfigFactory._get_train_fold_dir(env, root_dir)
        data_dir = input_dir / "shopee-product-matching"
        s_data_dir = ConfigFactory._get_s_data_dir(env, input_dir)
        if env != EnvEnum.KAGGLE:
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        return DirConfig(
            root_dir=root_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_out_dir=checkpoint_out_dir,
            dataset_dir=dataset_dir,
            data_dir=data_dir,
            s_data_dir=s_data_dir,
            yamls_dir=root_dir / "src" / "kaggle_shopee" / "yamls"
            if env != EnvEnum.KAGGLE
            else Path(
                "/kaggle/working/kaggle-shopee-6th-place-solution/src/kaggle_shopee/yamls"
            ),
            train_images_dir=data_dir / "train_images",
            test_images_dir=data_dir / "test_images",
            additional_train_dir=additional_train_dir,
            train_fold_dir=train_fold_dir,
        )

    @staticmethod
    def _get_config_from_yaml_dict(
        env: EnvEnum,
        exp: str,
        yaml_dict: Dict,
        dir_config: DirConfig,
    ) -> Config:
        config = Config(
            name=yaml_dict["name"],
            env=env,
            exp=exp,
            seed=yaml_dict["seed"],
            is_submitting=False,
            title_tfidf_n_components=int(yaml_dict["title_tfidf_n_components"]),
            title_tfidf_reducer=ReducerEnum[yaml_dict["title_tfidf_reducer"]],
            dir_config=dir_config,
            cv_config=CvConfig(**yaml_dict["cv_config"]),
            dataset_config=DatasetConfig(**yaml_dict["dataset_config"]),
            transform_config=TransformConfig(
                train=[
                    TransformComponent(**t)
                    for t in yaml_dict["transform_config"]["train"]
                ],
                valid=[
                    TransformComponent(**t)
                    for t in yaml_dict["transform_config"]["valid"]
                ],
                inference=[
                    TransformComponent(**t)
                    for t in yaml_dict["transform_config"]["inference"]
                ],
            ),
            model_config=ModelConfig(**yaml_dict["model_config"]),
            loss_config=LossConfig(
                **yaml_dict["loss_config"] if "loss_config" in yaml_dict else {}
            ),
            pooling_config=PoolingConfig(
                **yaml_dict["pooling_config"] if "pooling_config" in yaml_dict else {}
            ),
            bert_pooling_config=BertPoolingConfig(
                **yaml_dict["bert_pooling_config"]
                if "bert_pooling_config" in yaml_dict
                else {}
            ),
            optimizer_config=OptimizerConfig(**yaml_dict["optimizer_config"]),
            scheduler_config=SchedulerConfig(**yaml_dict["scheduler_config"]),
            train_config=TrainConfig(**yaml_dict["train_config"]),
            met_config=MetricLearningConfig(**yaml_dict["met_config"]),
            mlflow_config=MlflowConfig(
                **yaml_dict["mlflow_config"] if "mlflow_config" in yaml_dict else {}
            ),
            inference_config=InferenceConfig(**yaml_dict["inference_config"]),
        )
        if "epoch_configs" in yaml_dict["inference_config"]:
            config.inference_config.epoch_configs = [
                InferenceEpochConfig(**e)
                for e in yaml_dict["inference_config"]["epoch_configs"]
            ]
        if "concat_configs" in yaml_dict["inference_config"]:
            config.inference_config.concat_configs = [
                InferenceConcatConfig(**e)
                for e in yaml_dict["inference_config"]["concat_configs"]
            ]
        return config

    @staticmethod
    def get_config_from_yaml_str(
        yaml_str: str, env: Optional[EnvEnum] = None, verbose: bool = True
    ):
        if verbose:
            print(yaml_str)
        yaml_dict = yaml.load(yaml_str, Loader=yaml.Loader)
        exp = yaml_dict["exp"]
        if env is None:
            env = ConfigFactory._get_env()
        dir_config = ConfigFactory._get_dir_config(env, exp)
        config = ConfigFactory._get_config_from_yaml_dict(
            env, exp, yaml_dict, dir_config
        )
        return config

    @staticmethod
    def get_config_from_yaml_file(
        exp: str, env: Optional[EnvEnum] = None, verbose: bool = True
    ):
        if env is None:
            env = ConfigFactory._get_env()
            print("env:", env)
        dir_config = ConfigFactory._get_dir_config(env, exp)
        filepath = str(dir_config.yamls_dir / f"{exp}.yaml")
        with open(filepath, "rb") as f:
            yaml_dict = yaml.safe_load(f)
        if verbose:
            with open(filepath, "r") as f:
                print(f"=============== {exp}.yaml =================")
                print(f.read())
                print("==============================================")
        config = ConfigFactory._get_config_from_yaml_dict(
            env, exp, yaml_dict, dir_config
        )
        return config
