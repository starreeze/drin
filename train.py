from __future__ import annotations
from common import args
from common.args import *
from common.loss_metric import *
import torch
import lightning as pl

if args.model_type == "baseline":
    from baseline import data as data_module, model as model_module
elif args.model_type == "drgcn":
    from drgcn import data as data_module, model as model_module


class MELModel(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.metrics = torch.nn.ModuleList([TopkAccuracy(k).to("cuda") for k in metrics_topk]).to("cuda")
        self.loss = TripletLoss(triplet_margin)
        self.model = model

    def forward(self, batch):
        return self.model(batch)

    def _forward_step(self, batch, batch_idx):
        log_str = f" {batch_idx}\t"
        y = batch[-1]
        y_hat = self(batch[:-1])
        loss = self.loss(y, y_hat)
        log_str += f"loss: {float(loss):.5f}\t"
        for k, metric in zip(metrics_topk, self.metrics):
            metric.update(y_hat, y)  # type: ignore
            log_str += f"top-{k}: {float(metric.compute()):.5f}\t"  # type: ignore
        print(log_str, end="\r")
        return loss

    def training_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


class EpochLogger(pl.Callback):
    def __init__(self, start_epoch, max_epoch) -> None:
        super().__init__()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

    def epoch_start(self, trainer: pl.Trainer, model: MELModel, type: str):
        for metric in model.metrics:
            metric.reset()  # type: ignore
        status_str = f"\n========== Epoch "
        status_str += f"{trainer.current_epoch + self.start_epoch + 1}/{self.max_epoch} - {type}"
        print(status_str)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        self.epoch_start(trainer, pl_module, "training")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        print("")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        self.epoch_start(trainer, pl_module, "validating")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        print("")

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        self.epoch_start(trainer, pl_module, "testing")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        print("")


def main():
    print("=============== parameters ===============")
    for arg in dir(args):
        if arg.startswith("__"):
            continue
        print(arg, getattr(args, arg), sep=" = ")
    pl.seed_everything(seed)
    datasets = data_module.create_datasets()
    model = MELModel(model_module.Model())
    try:
        for i in range(num_epoch // test_epoch_interval):
            trainer = pl.Trainer(
                enable_checkpointing=False,
                max_epochs=test_epoch_interval,
                accelerator="gpu",
                devices=1,
                enable_progress_bar=False,
                callbacks=EpochLogger(i * test_epoch_interval, num_epoch),
                enable_model_summary=False,
            )
            trainer.fit(model, datasets[0], datasets[1])
            trainer.test(model, datasets[2])
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
