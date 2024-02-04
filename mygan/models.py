from typing import Any, Self

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision.utils import make_grid

from mygan.noise import gen_noise


def init_weights(m: nn.Module) -> None:
  if any(isinstance(m, t) for t in (nn.BatchNorm2d, nn.Conv2d, nn.ConvTranspose2d)):
    torch.nn.init.normal_(m.weight, mean=0, std=0.02)
    torch.nn.init.constant_(m.weight, val=0)


class Generator(nn.Module):
  @staticmethod
  def block(in_dim: int, out_dim: int, stride: int = 2, pad: int = 1, bn: bool = True) -> nn.Sequential:
    layers = [nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=4, stride=stride, padding=pad)]
    if bn:
      layers.append(nn.BatchNorm2d(out_dim))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)

  def __init__(self: Self, z_dim: int = 64, d_dim: int = 16) -> None:
    super().__init__()
    # check if z_dim is a power of 2
    # TODO generate the blocks using this
    # assert (z_dim & (z_dim - 1) == 0) and z_dim != 0
    self.z_dim = z_dim
    self.d_dim = d_dim
    # (n-1)*stride-2*padding + ks
    # n = width or height
    # ks = kernel size
    # we begin with a 1x1 image with z_dim number of channels (consider 200)
    self.model = nn.Sequential(
      # (n-1)*stride-2*padding + ks = (1-1)*1-2*0+4=4
      # 4x4(ch: 200 -> 512)
      Generator.block(z_dim, d_dim * 32, stride=1, pad=0),
      # (n-1)*stride-2*padding + ks = (4-1)*2-2*1+4=3*2-2+4=6-2+4=8
      # 8x8(ch: 512 -> 256)
      Generator.block(d_dim * 32, d_dim * 16),
      # (n-1)*stride-2*padding + ks = (8-1)*2-2*1+4=14-2+4=16
      # 16x16(ch: 256 -> 128)
      Generator.block(d_dim * 16, d_dim * 8),
      # (n-1)*stride-2*padding + ks = (16-1)*2-2*1+4=30-2+4=32
      # 32x32(ch: 128 -> 64)
      Generator.block(d_dim * 8, d_dim * 4),
      # (n-1)*stride-2*padding + ks = (32-1)*2-2*1+4=62-2+4=64
      # 64x64(ch: 64 -> 32)
      Generator.block(d_dim * 4, d_dim * 2),
      # (n-1)*stride-2*padding + ks = (64-1)*2-2*1+4=14-2+4=128
      # 128x128(ch: 32 -> 3)
      nn.ConvTranspose2d(in_channels=d_dim * 2, out_channels=3, kernel_size=4, stride=2, padding=1),
      nn.Tanh(),
    )

  def forward(self: Self, noise: torch.Tensor) -> torch.Tensor:
    x = noise.view(len(noise), self.z_dim, 1, 1)  # batch_size x n_channels x w x h = 128x200x1x1
    return self.model(x)


class Discriminator(nn.Module):
  @staticmethod
  def block(in_dim: int, out_dim: int, instance_norm: bool = True, leaky: bool = True) -> nn.Module:
    layers = [nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)]
    if instance_norm:
      layers.append(nn.InstanceNorm2d(out_dim))
    if leaky:
      layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    else:
      layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

  def __init__(self: Self, d_dim: int = 16) -> None:
    super().__init__()
    self.d_dim = d_dim
    # new size = (n + 2 * pad - ks) // stride + 1
    # n = width or height
    # pad = padding
    # ks = kernel size
    # we begin with a 128x128x3 image
    self.model = nn.Sequential(
      # (n + 2 * pad - ks) // stride + 1 = (128 + 2 * 1 - 4) // 2 + 1 = (128 - 2) // 2 + 1 = 63 + 1 = 64
      Discriminator.block(3, d_dim),
      # (n + 2 * pad - ks) // stride + 1 = (64 + 2 * 1 - 4) // 2 + 1 = (64 - 2) // 2 + 1 = 31 + 1 = 32
      Discriminator.block(d_dim, d_dim * 2),
      # (n + 2 * pad - ks) // stride + 1 = (32 + 2 * 1 - 4) // 2 + 1 = (32 - 2) // 2 + 1 = 15 + 1 = 16
      Discriminator.block(d_dim * 2, d_dim * 4),
      # (n + 2 * pad - ks) // stride + 1 = (16 + 2 * 1 - 4) // 2 + 1 = (16 - 2) // 2 + 1 = 7 + 1 = 8
      Discriminator.block(d_dim * 4, d_dim * 8),
      # (n + 2 * pad - ks) // stride + 1 = (8 + 2 * 1 - 4) // 2 + 1 = (8 - 2) // 2 + 1 = 3 + 1 = 4
      Discriminator.block(d_dim * 8, d_dim * 16),
      # (n + 2 * pad - ks) // stride + 1 = (4 + 0 * 1 - 4) // 1 + 1 = (4 - 4) // 2 + 1 = 0 + 1 = 1
      nn.Conv2d(in_channels=d_dim * 16, out_channels=1, kernel_size=4, stride=1, padding=0),
    )

  def forward(self: Self, image: torch.Tensor) -> torch.Tensor:
    # input: batch x channels x w x h
    # image: 128 x 3 x 128 x 128
    crit_pred = self.model(image)  # 128 x 1 x 1 x 1
    return crit_pred.view(len(crit_pred), -1)


class GAN(LightningModule):
  def __init__(self: Self, config: dict[str, Any]) -> None:
    super().__init__()
    self.save_hyperparameters()
    self.z_dim = config["z_dim"]
    self.d_dim = config["d_dim"]
    self.lr = config["lr"]
    self.betas = config["betas"]
    self.critic_cycles = config["critic_cycles"]
    self.val_size = config["val_size"]
    self.gradient_penalty = config["gradient_penalty"]
    self.loss = config["loss"]
    self.automatic_optimization = False
    self.gen = Generator(z_dim=self.z_dim, d_dim=self.d_dim)
    self.disc = Discriminator(d_dim=self.d_dim)
    self.val_noise = gen_noise(config["val_size"], self.z_dim)

  def forward(self: Self, noise: torch.Tensor) -> torch.Tensor:
    return self.gen(noise)

  # gradient penalty calculation
  def compute_gradient_penalty(self: Self, real: torch.Tensor, fake: torch.Tensor, gamma: float = 10):
    batch_size = len(real)
    alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True).type_as(real)
    mix_images = real * alpha + fake * (1 - alpha)  # 128 x 3 x 128 x 128
    mix_scores = self.disc(mix_images)

    # 128 x 3 x 128 x 128
    gradient = torch.autograd.grad(
      inputs=mix_images,
      outputs=mix_scores,
      grad_outputs=torch.ones_like(mix_scores),
      retain_graph=True,
      create_graph=True,
    )[0]
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    return gamma * ((gradient_norm - 1) ** 2).mean()

  def configure_optimizers(self: Self):
    g_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=self.betas)
    d_opt = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=self.betas)
    return g_opt, d_opt

  def training_step(self: Self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
    real_imgs, _ = batch
    batch_size = len(real_imgs)
    g_opt, d_opt = self.optimizers()

    # critic
    mean_crit_loss = 0
    for _ in range(self.critic_cycles):
      crit_loss = self.compute_loss_disc(real_imgs)
      d_opt.zero_grad()
      self.manual_backward(crit_loss)
      d_opt.step()
      mean_crit_loss += crit_loss.item() / self.critic_cycles

    # generator
    gen_loss = self.compute_loss_gen(batch_size)
    g_opt.zero_grad()
    self.manual_backward(gen_loss)
    # gen_loss.backward()
    g_opt.step()

    self.log_dict({"Generator Loss/Train": gen_loss, "Mean Critic Loss/Train": mean_crit_loss})

    if batch_idx % 50 == 0:
      with torch.no_grad():
        fake = self.gen(self.val_noise.type_as(real_imgs))
        fake = fake.detach().cpu()
        grid = make_grid(fake, nrow=int(len(fake) ** 0.5))
        self.logger.log_image(key="Samples", images=[grid], step=self.current_epoch)

  def compute_loss_gen(self: Self, batch_size: int):
    noise = gen_noise(batch_size, self.gen.z_dim).type_as(self.gen.model[0][0].weight)
    fake = self.gen(noise)
    pred = self.disc(fake)
    if self.loss == "w":
      return -pred.mean()
    return nn.functional.binary_cross_entropy_with_logits(pred, torch.ones_like(pred))

  def compute_loss_disc(self: Self, real_imgs: torch.Tensor):
    batch_size = len(real_imgs)
    noise = gen_noise(batch_size, self.gen.z_dim).type_as(real_imgs)
    fake = self.gen(noise)
    disc_fake_pred = self.disc(fake.detach())
    disc_real_pred = self.disc(real_imgs)
    gp = self.compute_gradient_penalty(real_imgs, fake.detach()) if self.gradient_penalty else 0

    if self.loss == "w":
      disc_loss = disc_fake_pred.mean() - disc_real_pred.mean()
    else:
      disc_fake_targets = torch.zeros_like(disc_fake_pred)
      disc_fake_loss = nn.functional.binary_cross_entropy_with_logits(disc_fake_pred, disc_fake_targets)

      disc_real_targets = torch.ones_like(disc_real_pred)
      disc_real_loss = nn.functional.binary_cross_entropy_with_logits(disc_real_pred, disc_real_targets)

      disc_loss = (disc_fake_loss + disc_real_loss) / 2

    return disc_loss + gp
