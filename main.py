from typing import List
import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from lightning_classes import utils
from lightning.pytorch.loggers import Logger

@hydra.main(config_path="config/", config_name="config.yaml", version_base = "1.2")
def main(cfg: DictConfig):
    # logger and callbacks setup
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    # instantiate datamodule, model and trainer components
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks= callbacks)
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
