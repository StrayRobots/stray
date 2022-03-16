import click
import torch
from stray.training.model import KeypointTrainModule


@click.command()
@click.argument('checkpoint')
def main(checkpoint):
    model = KeypointTrainModule.load_from_checkpoint(checkpoint)
    script = model.to_torchscript()
    torch.jit.save(script, "model.pt")

if __name__ == "__main__":
    main()