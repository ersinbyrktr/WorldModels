# utils/collate.py
from torch.nn.utils.rnn import pad_sequence
import torch

def pad_collate(batch):
    xs, ys, lens = zip(*batch)               # dataset always returns 3-tuple
    lens   = torch.as_tensor(lens, dtype=torch.long)
    xb_pad = pad_sequence(xs, batch_first=True)   # (B, Tmax, C)
    yb_pad = pad_sequence(ys, batch_first=True)
    return xb_pad, yb_pad, lens
