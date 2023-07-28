from __future__ import annotations
from common import args
from common.args import *
from common.utils import *
from datetime import datetime
import torch
import lightning as pl

if args.model_type == "ghmfc":
    from baselines import data as data_module, ghmfc as model_module
elif args.model_type == "melhi":
    from baselines import data as data_module, melhi as model_module
elif args.model_type == "drin":
    from drin import data as data_module, model as model_module

if output_test_result:
    result_file = open('test-result.txt', 'w')


class MELModel(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.metrics = torch.nn.ModuleList([TopkAccuracy(k).to(use_device) for k in metrics_topk]).to(use_device)
        self.loss = TripletLoss(triplet_margin)
        self.model = model

    def forward(self, batch):
        return self.model(batch)

    def _forward_step(self, batch, batch_idx, type):
        log_str = f" {batch_idx}\t"
        y = batch[-1]
        y_hat = self(batch[:-1])
        loss = self.loss(y, y_hat)
        log_str += f"loss: {float(loss):.5f}\t"
        for k, metric in zip(metrics_topk, self.metrics):
            metric.update(y_hat, y)  # type: ignore
            log_str += f"top-{k}: {float(metric.compute()) / (1-acc_correction[type]):.5f}\t"  # type: ignore
        print(log_str, end="\r")
        if output_test_result and type == 2:
            for i, sample in enumerate(y_hat.to('cpu').tolist()):
                result_file.write(f'{i + batch_idx * batch_size}:\t{sample}\n{y[i]}\n')
                result_file.flush()
        return loss

    def training_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, 0)

    def validation_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, 1)

    def test_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, 2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


class EpochLogger(pl.Callback):
    def __init__(self, start_epoch, max_epoch) -> None:
        super().__init__()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        if profiling:
            self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("log/profiler"),
                record_shapes=True,
                with_stack=True,
            )

    def epoch_start(self, trainer: pl.Trainer, model: MELModel, type: str):
        for metric in model.metrics:
            metric.reset()  # type: ignore
        status_str = f"\n***** Epoch "
        status_str += f"{trainer.current_epoch + self.start_epoch + 1}/{self.max_epoch} - {type} - {datetime.now()}"
        print(status_str)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        self.epoch_start(trainer, pl_module, "training")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        print("")
        if profiling:
            self.prof.stop()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        self.epoch_start(trainer, pl_module, "validating")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        print("")

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        self.epoch_start(trainer, pl_module, "testing")
        if output_test_result:
            result_file.write('==========  Test ==========\n')

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: MELModel) -> None:
        print("")

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if profiling:
            self.prof.start()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int
    ) -> None:
        if profiling:
            self.prof.step()


def create_trainer(start_epoch) -> pl.Trainer:
    return pl.Trainer(
        num_sanity_val_steps=2 if args.debug else 0,
        enable_checkpointing=False,
        max_epochs=test_epoch_interval,
        accelerator="gpu" if use_device == "cuda" else None,
        devices=1,
        enable_progress_bar=False,
        callbacks=EpochLogger(start_epoch, num_epoch),
        enable_model_summary=False,
    )


def main():
    print("=============== parameters ===============")
    for arg in dir(args):
        if arg.startswith("__"):
            continue
        value = getattr(args, arg)
        if type(value) == str:
            value = "'" + value + "'"
        print(arg, value, sep=" = ")
    pl.seed_everything(seed)
    datasets = data_module.create_datasets()
    model = MELModel(model_module.Model())
    if test_only:
        trainer = create_trainer(0)
        trainer.test(model, datasets[2])
        return
    for i in range(num_epoch // test_epoch_interval):
        trainer = create_trainer(i * test_epoch_interval)
        trainer.fit(model, datasets[0], datasets[1])
        trainer.test(model, datasets[2])
    if output_test_result:
        result_file.close()
    print("Training completed")
    # if use_device == "cuda":
    #     x = torch.empty([10531 * 1024 * 1024], dtype=torch.uint8)
    #     time.sleep(60 * 60 * 24)


if __name__ == "__main__":
    main()
