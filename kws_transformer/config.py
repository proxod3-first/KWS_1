from dataclasses import dataclass
from typing import Literal

@dataclass
class Model:
    name: str
    layers: int
    heads: int
    mlp_dim: int
    num_output_classes: int
    embedding_dim: int
    norm_type: Literal["postnorm", "prenorm"]
    dropout: float

@dataclass
class Data:
    time_window: int
    frequency: int
    patch_size_t: int
    patch_size_f: int

@dataclass
class Training:
    batch: int
    lr: float
    save_dir_tr: str
    save_dir_wandb: str
    model_path: str
    epochs: int

@dataclass
class Dataset:
    sample_rate: int
    destination: str

@dataclass
class MFCC_Set:
    sample_rate: int
    n_mfcc: int
    n_mels: int
    n_fft: int
    hop_length: int

@dataclass
class Params:
    model: Model
    data: Data
    training: Training
    dataset: Dataset
    mfcc_settings: MFCC_Set