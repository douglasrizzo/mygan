import random
import zipfile
from enum import Enum
from pathlib import Path
from typing import Self

import requests
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder
from tqdm import tqdm

PATH_DATASETS = Path("data/")


def download_with_progress(url: str | Path, filepath: Path) -> None:
  """Download a file from the given URL to the specified filepath while displaying the download progress.

  From https://stackoverflow.com/a/37573701/1245214.

  Args:
      url (str | Path): The URL of the file to download.
      filepath (Path): The filepath to save the downloaded file.
  """
  # Sizes in bytes.
  response = requests.get(url, stream=True)
  total_size = int(response.headers.get("content-length", 0))
  block_size = 1024

  with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar, filepath.open("wb") as file:
    for data in response.iter_content(block_size):
      progress_bar.update(len(data))
      file.write(data)


def _init_loader_seed(worker_id: int) -> None:
  """Initialize the seed for the loader.

  Parameters:
      worker_id (int): The ID of the worker.
  """
  random.seed(random.getstate()[1][0] + worker_id)


class CelebASource(Enum):
  """Multiple sources for downloading the CelebA dataset."""

  KAGGLE = "https://storage.googleapis.com/kaggle-data-sets/29561/37705/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240203%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240203T143127Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=946521c778df7e528e7fa3e335b0fc4c159d74ae5f08d593d0d331b6aea4fe23b097f7d101928e90acd585b3514d445e67459489aa15600debe5a0e612d2c6adf8de967e5e93de03472043dfc2d7a6c2580cccde78c8fc3197a03c19c439066273591aa66f246a22361c82c49ddd6c8360ee1384077334c9c103a87c7390b50ff299a954690e688739a5fbedef186bf95ee6760e592f4647b4bafae2de8ebbc43c45a98e956fde1673b2ebd978d8c1e491a98568bc7828c75c75146febb156919d2adce9e4fb4e87b5fb552f3b37bc4a4ebaadcdcbfd0e70cb4ec3329e3394b6213dfddeb2ee037a54a94aefeea533dc69dfd7f1450f93c4b05f314b29109b65"
  GOOGLE_DRIVE = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"


class CelebADataModule(LightningDataModule):
  """CelebFaces Attributes Dataset (CelebA).

  A large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The
  images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large
  quantities, and rich annotations, including:

  - 10,177 number of identities,
  - 202,599 number of face images, and
  - 5 landmark locations, 40 binary attributes annotations per image.

  The dataset can be employed as the training and test sets for the following computer vision tasks: face attribute
  recognition, face recognition, face detection, landmark (or facial part) localization, and face editing & synthesis.
  """

  def __init__(
    self: Self,
    batch_size: int,
    num_workers: int = 0,
    source: CelebASource = CelebASource.KAGGLE,
    data_dir: Path | str = PATH_DATASETS,
    img_size: int = 128,
  ) -> None:
    """Instantiate the CelebFaces Attributes data module. Download it if it does not exist locally.

    Args:
        self (Self): The object instance.
        path (pl.Path | str): The path to download or find the dataset.
        source (CelebASource): The source to download the dataset from.
        size (int): The size of the images.
        lim (int): The limit of items to consider.
    """
    super().__init__()
    self.sizes = [img_size, img_size]
    self.data_dir = Path(data_dir).absolute()
    self.download_file_path = self.data_dir / "img_align_celeba.zip"
    self.images_path = self.data_dir / "img_align_celeba"
    self.source_url = source.value
    self.batch_size = batch_size
    self.num_workers = num_workers
    items, labels = [], []
    for data in list(data_dir.glob("**/*.jpg")):
      # path './data/celeba/img_align_celeba'
      # data '114568.jpg'
      items.append(data)
      labels.append(int(data.stem))

    self.train_split = ImageFolder(
      root=self.images_path,
      transform=transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]),
    )

  def prepare_data(self: Self) -> None:
    """Downloads a file from a given URL and saves it to a specified file path."""
    if not self.data_dir.exists():
      self.data_dir.mkdir(parents=True)

    download_file_path = self.data_dir / "img_align_celeba.zip"
    images_path = self.data_dir / "img_align_celeba"

    if not images_path.exists():
      if not download_file_path.exists():
        download_with_progress(self.source_url, download_file_path)
      with zipfile.ZipFile(download_file_path, "r") as ziphandler:
        ziphandler.extractall(self.data_dir)

  def train_dataloader(self: Self) -> DataLoader:
    """Creates and returns a DataLoader for the training data.

    Args:
        self (Self): The instance of the class.

    Returns:
        DataLoader: The DataLoader object for training data.
    """
    return DataLoader(
      self.train_split,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      worker_init_fn=_init_loader_seed,
    )


class MNISTDataModule(LightningDataModule):
  def __init__(self, batch_size: int, num_workers: int, data_dir: str = PATH_DATASETS):
    super().__init__()
    self.batch_size = batch_size
    self.data_dir = data_dir
    self.num_workers = num_workers
    self.transform = transforms.ToTensor()
    self.dims = (1, 28, 28)
    self.num_classes = 10

  def prepare_data(self):
    # download
    MNIST(self.data_dir, train=True, download=True)

  def setup(self, stage=None):
    # Assign train/val datasets for use in dataloaders
    if stage == "fit" or stage is None:
      self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)

  def train_dataloader(self):
    return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
