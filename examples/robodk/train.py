import click
import torch
import os
import pytorch_lightning as pl

from stray.training.data import KeypointSceneData
from stray.training.model import KeypointTrainModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

@click.command()
@click.argument('dataset')
@click.option('--eval-folder', type=str, required=True)
@click.option('--primitive', type=click.Choice(['keypoints', 'rectangle']))
@click.option('--num-instances', type=int, default=1)
@click.option('--batch-size', type=int, default=4)
@click.option('--num-workers', type=int, default=1)
@click.option('--num-epochs', type=int, default=1)
@click.option('--lr', type=float, default=1e-4)
@click.option('--fp16', is_flag=True)
@click.option('--tune', is_flag=True)
@click.option('--restore', type=str)
@click.option('--logs', default="./train_logs", type=str)
def train(dataset, eval_folder, primitive, num_instances, batch_size, num_workers, num_epochs,
        lr, fp16, tune, restore, logs):

    kp_config = dict(primitive=primitive, num_instances=num_instances)
    if primitive == "keypoints":
        num_keypoints = num_instances
    elif primitive == "rectangle":
        num_keypoints = 4*num_instances
    else:
        raise ValueError(f"Incorrect primitive {primitive}")

    if restore:
        model = KeypointTrainModule.load_from_checkpoint(restore, num_keypoints=num_keypoints, lr=lr)
    else:
        model = KeypointTrainModule(num_keypoints=num_keypoints, lr=lr)

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
        'logger': loggers.TensorBoardLogger(logs),
        'max_epochs': num_epochs
    }
    if fp16:
        config['precision'] = 16

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="kpdetection-{epoch:02d}-{val_loss:.2f}",
        save_top_k=10,
        mode="min",
    )

    train = pl.Trainer(**config, callbacks=[checkpoint_callback], check_val_every_n_epoch=10)

    scenes = [os.path.join(dataset, p) for p in os.listdir(dataset)]
    train_paths = [scene for scene in scenes if (os.path.isdir(scene))]
    eval_paths = [os.path.join(eval_folder, folder) for folder in os.listdir(eval_folder) if os.path.isdir(os.path.join(eval_folder, folder) )]

    datamodule = KeypointSceneData(train_paths, eval_paths, batch_size, batch_size, num_workers, kp_config, (640, 480), (80, 60))

    if tune:
        train.tune(model, datamodule=datamodule)

    train.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train()
