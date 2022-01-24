from stray.models.keypoint_heatmap.train import KeypointHetmapTrainModule
from stray.data.keypoint_heatmap import KeypointHeatmapDataModule
import pytorch_lightning as pl
import click
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


@click.command()
@click.argument('model-type', type=click.Choice(["keypoint-heatmap"]))
@click.option('--train-data', type=str, required=True)
@click.option('--eval-data', type=str, required=True)
@click.option('--batch-size', type=int, default=2)
@click.option('--num-workers', type=int, default=1)
@click.option('--num-epochs', type=int, default=1)
@click.option('--width', '-w', type=int, default=480)
@click.option('--height', '-h', type=int, default=640)
@click.option('--num-heatmaps', type=int, default=4) #Only for heatmap detection
@click.option('--lr', type=float, default=1e-3)
@click.option('--fp16', is_flag=True)
@click.option('--tune', is_flag=True)
@click.option('--checkpoint', type=str)
def main(model_type, train_data, eval_data, batch_size, num_workers, num_epochs, width, height, num_heatmaps, lr, fp16, tune, checkpoint):

    train_scene_paths = [os.path.join(train_data, p) for p in os.listdir(train_data) if os.path.isdir(os.path.join(train_data, p))]
    eval_scene_paths = [os.path.join(eval_data, p) for p in os.listdir(eval_data) if os.path.isdir(os.path.join(eval_data, p))]

    if model_type == "keypoint-heatmap":
        train_module = KeypointHetmapTrainModule
        data_module = KeypointHeatmapDataModule
    else:
        raise ValueError(f"Unknown model type {model_type}")

    if checkpoint:
        model = train_module.load_from_checkpoint(checkpoint, lr=lr, num_heatmaps=num_heatmaps)
    else:
        model = train_module(lr, num_heatmaps)
        

    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    config = {
        'gpus': gpus,
        'accumulate_grad_batches': {
            0: 1,
            num_epochs // 4: 2,
            num_epochs // 2: 4
        },
        'max_epochs': num_epochs
    }
    if fp16:
        config['precision'] = 16

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename=model_type+"-detection-{epoch:02d}-{val_loss:.2f}",
        save_top_k=10,
        mode="min",
    )

    train = pl.Trainer(**config, callbacks=[checkpoint_callback], check_val_every_n_epoch=1, profiler="advanced")

    data = data_module(train_scene_paths, eval_scene_paths, batch_size, batch_size, num_workers, num_heatmaps, (width, height), (60, 80))

    if tune:
        train.tune(model, datamodule=data)

    train.fit(model, datamodule=data)

if __name__ == "__main__":
    main()
