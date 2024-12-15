"""Implements a common pytorch lightning schema for all segmentation models."""


from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class SegmentationExperiment(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        multitask_loss: dict[str, nn.Module],
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        optimizer_kwargs: Optional[dict[str, float]] = None,
        scheduler_kwargs: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """"""
        super().__init__()
        self.model = model
        self.heads = model.heads
        self.aux_key = model.aux_key
        self.inst_key = model.inst_key

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.criterion = multitask_loss

        self._validate_branch_args()
        self.save_hyperparameters(ignore="model")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(x)

    def step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute the loss for one batch.

        Parameters
        ----------
            batch : Dict[str, torch.Tensor]
                The batch of data.
            batch_idx : int
                The batch index.

        Returns
        -------
            Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
                The loss, the soft masks and the targets.
        """
        soft_masks = self.model(batch["image"])
        targets = {k: val for k, val in batch.items() if k != "image"}

        loss = self.criterion(
            yhats=soft_masks,
            targets=targets,
            mask=targets["dist"],  # we will use the distance map as a mask
        )

        return loss, soft_masks, targets

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step."""
        # forward backward pass
        loss, _, _ = self.step(batch)

        # log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        # forward pass
        loss, _, _ = self.step(batch)

        # log the loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        # forward pass
        loss, _, _ = self.step(batch)

        # log the loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self) -> List[optim.Optimizer]:
        """Configure the optimizers for the model."""
        opt = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        sch = self.scheduler(opt, **self.scheduler_kwargs)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "name": "train/lr",  # we can log the lr if needed
                "scheduler": sch,
                "monitor": "val_loss",
            },
        }

    def _validate_branch_args(self) -> None:
        """Check that there are no conflicting decoder branch args."""
        lk = set([k.split("_")[0] for k in self.criterion.keys()])
        dk = set(self.model._get_inner_keys(self.model.heads))
        has_same_keys = lk == dk

        if not has_same_keys:
            raise ValueError(
                "Got mismatching keys for branch dict args. "
                f"Multitask loss branches: {lk}. "
                f"Decoder branches: {dk}. "
                f"(`branch_metrics` can be None)"
            )