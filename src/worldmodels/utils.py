import random

import numpy as np
import torch
from torchvision import utils as vutils


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_grid(t: torch.Tensor, fp: str, nrow: int = 8):
    vutils.save_image(t.clamp(0, 1).cpu(), fp, nrow=nrow)
