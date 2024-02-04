import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from mygan.datasets import CelebADataModule, MNISTDataModule
from mygan.models import GAN

WANDB_PROJECT_NAME = "wgan_celeba"


def main() -> None:
  """Train a DCGAN model on MNIST."""
  parser = argparse.ArgumentParser(
    "DCGAN training script.",
    description="Train a DCGAN model on MNIST to generate number images.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  group = parser.add_argument_group("Training parameters")
  group.add_argument(
    "--batch_size",
    type=int,
    default=1024,
    help="Number of training examples utilized in one iteration.",
  )
  group.add_argument(
    "--epochs",
    type=int,
    default=10000,
    help="Maximum number of training epochs.",
  )
  group.add_argument(
    "--precision",
    type=int,
    default=16,
    help="Floating point precision to use.",
  )
  group.add_argument(
    "--critic_cycles",
    type=int,
    default=5,
    help="Number of training cycles for the critic, w.r.t. the generator.",
  )
  group.add_argument(
    "--wandb",
    dest="wandb",
    action="store_true",
    help="Log results to wandb in addition to tensorboard.",
  )
  group = parser.add_argument_group("Loss parameters")
  group.add_argument(
    "--loss",
    type=str,
    choices=["w", "bce"],
    default="w",
    help="Which loss to use. 'w' for Wasserstein, otherwise binary cross-entropy.",
  )
  group.add_argument(
    "--no-gp",
    action="store_false",
    dest="gp",
    help="Disable the usage of gradient penalty on the loss.",
  )

  group = parser.add_argument_group("Dataset parameters")
  group.add_argument(
    "--dataset",
    type=str,
    choices=["celeba", "mnist"],
    default="mnist",
    help="Dataset to use for training",
  )

  group = parser.add_argument_group("Model parameters")
  group.add_argument(
    "--z_dim",
    type=int,
    default=200,
    help="Number of embedding dimensions",
  )
  group.add_argument(
    "--d_dim",
    type=int,
    default=16,
    help="Number of ??? dimensions",
  )

  group = parser.add_argument_group("Optimizer parameters")
  group.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate",
  )
  group.add_argument(
    "--betas",
    nargs=2,
    type=float,
    default=[0.5, 0.9],
    help="Beta parameters for Adam optimizer",
  )

  args = parser.parse_args()

  model = GAN({
    "z_dim": args.z_dim,
    "d_dim": args.d_dim,
    "lr": args.lr,
    "betas": tuple(args.betas),
    "critic_cycles": args.critic_cycles,
    "val_size": 25,
    "loss": args.loss,
    "gradient_penalty": args.gp,
  })

  datamodule = (
    CelebADataModule(batch_size=args.batch_size, num_workers=7)
    if args.dataset == "celeba"
    else MNISTDataModule(batch_size=args.batch_size, num_workers=7)
  )
  trainer = Trainer(
    logger=WandbLogger(project=WANDB_PROJECT_NAME),
    accelerator="gpu",
    max_epochs=args.epochs,
    val_check_interval=0.05,
    precision=16,
    enable_progress_bar=True,
    callbacks=[RichProgressBar()],
  )
  trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
  main()
