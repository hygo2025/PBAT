
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import (
    LR_SCHEDULER_REGISTRY,
    LRSchedulerTypeTuple,
    OPTIMIZER_REGISTRY,
    LightningCLI,
)
from src.model import RecModel
from src.datamodule import RecDataModule


class PBATLightningCLI(LightningCLI):
    def _add_arguments(self, parser):
        # PL 1.5.x expects schedulers to derive from _LRScheduler, which changed in torch 2.x.
        self.add_default_arguments_to_parser(parser)
        self.add_core_arguments_to_parser(parser)
        self.add_arguments_to_parser(parser)
        if not parser._optimizers:
            parser.add_optimizer_args(OPTIMIZER_REGISTRY.classes)
        if not parser._lr_schedulers:
            compatible_lr_schedulers = tuple(
                cls for cls in LR_SCHEDULER_REGISTRY.classes if issubclass(cls, LRSchedulerTypeTuple)
            )
            if compatible_lr_schedulers:
                parser.add_lr_scheduler_args(compatible_lr_schedulers)
        self.link_optimizers_and_lr_schedulers(parser)


def cli_main():
    cli = PBATLightningCLI(RecModel, RecDataModule, save_config_overwrite=True)

if __name__ == '__main__':
    cli_main()

