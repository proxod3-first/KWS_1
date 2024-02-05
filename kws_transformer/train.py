from kws.model import ViT_Lightning, ConfMatrixLogging
from kws.data import Audio_DataModule, NewEra_AudioDataModule
from kws.preprocessing_data.preproccesing import KNOWN_COMMANDS

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import hydra
from hydra.core.config_store import ConfigStore
from config import Params

cs = ConfigStore.instance()
cs.store(name="params", node=Params)

def get_labels():
    """
    backward - 0, bed - 1, bird - 2, cat - 3, dog - 4,
    down - 5, eight - 6, five - 7, follow - 8, forward - 9,
    four - 10, go - 11, happy - 12, house - 13, learn - 14,
    left - 15, marvin - 16, nine - 17, no - 18, off - 19,
    on - 20, one - 21, right - 22, seven - 23, sheila - 24,
    six - 25, stop - 26, three - 27, tree - 28, two - 29,
    up - 30, visual - 31, wow - 32, yes - 33, zero - 34
    """
    return [
        'backward', 'bed', 'bird', 'cat', 'dog', 
        'down', 'eight', 'five', 'follow', 'forward', 
        'four', 'go', 'happy', 'house', 'learn', 
        'left', 'marvin', 'nine', 'no', 'off', 
        'on', 'one', 'right', 'seven', 'sheila', 
        'six', 'stop', 'three', 'tree', 'two', 
        'up', 'visual', 'wow', 'yes', 'zero'
    ]

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    # dm = Audio_DataModule(
    #     data_destination=cfg.dataset.destination,
    #     batch_size=cfg.training.batch,
    #     audio_rate=cfg.dataset.sample_rate,
    #     labels=get_labels()
    # )

    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    dm = NewEra_AudioDataModule(
        data_destination=cfg.dataset.destination,
        batch_size=cfg.training.batch,
        n_mfcc=cfg.mfcc_settings.n_mfcc,
        hop_length=cfg.mfcc_settings.hop_length
    )
    
    model = ViT_Lightning(
        time_window=cfg.data.time_window,
        frequency=cfg.mfcc_settings.n_mfcc,
        patch_size_t=cfg.data.patch_size_t,
        patch_size_f=cfg.data.patch_size_f,
        embed_dim=cfg.model.embedding_dim,
        num_classes=cfg.model.num_output_classes,
        depth=cfg.model.layers,
        num_heads=cfg.model.heads,
        mlp_dim=cfg.model.mlp_dim,
        norm_type=cfg.model.norm_type,
        lr=cfg.training.lr,
        qkv_bias=False,
        drop_rate=cfg.model.dropout,
        # -----
        type_of_scheduler = "ReduceOnPlateau", 
        patience_reduce = 5, 
        factor_reduce = 0.1, 
        # -----
        # type_of_scheduler = "OneCycleLR",
        # lr_coef_cycle = 1, 
        # total_num_of_epochs = cfg.training.epochs,
        #-----
        sample_rate = cfg.mfcc_settings.sample_rate, 
        n_mffc = cfg.mfcc_settings.n_mfcc, 
        n_mels = cfg.mfcc_settings.n_mels, 
        n_fft = cfg.mfcc_settings.n_fft, 
        hop_length=cfg.mfcc_settings.hop_length,
        previous_model = None,
        need_mfcc=False
    )

    wandb.login(key="dec2ee769ce2e455dd463be9b11767cf8190d658")
    wandb_log = WandbLogger(project="KWS", name="Final-h2-l12-pre-emb256/512-p20/8-rop(3e-4)", save_dir=working_dir + "/wandb")

    checkpoint = ModelCheckpoint(
        dirpath=working_dir + "/weights",
        save_top_k=3,
        monitor="val_loss"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    conf_matrix_logger = ConfMatrixLogging(KNOWN_COMMANDS)
    # early_stop = EarlyStopping(monitor="val_loss", patience=7)

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_log,
        callbacks=[checkpoint, lr_monitor, conf_matrix_logger],
        # fast_dev_run=5
    )
    trainer.fit(model=model, datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    L.seed_everything(1702)
    main()