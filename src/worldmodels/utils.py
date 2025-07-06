import torch
from torchvision import utils as vutils


def save_grid(t: torch.Tensor, fp: str, nrow: int = 8):
    vutils.save_image(t.clamp(0, 1).cpu(), fp, nrow=nrow)
