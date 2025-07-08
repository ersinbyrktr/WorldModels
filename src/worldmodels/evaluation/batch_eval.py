# evaluation/batch_eval.py
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

@torch.no_grad()
def evaluate(model, loader):
    """
    Average NLL per real frame, with padding support.
    """
    dev = next(model.parameters()).device
    model.eval()
    tot_nll, tot_frames = 0.0, 0

    for xb, yb, lens in loader:                # xb: (B,Tmax,C)
        xb, yb, lens = xb.to(dev), yb.to(dev), lens.to(dev)
        B, Tmax, Cin = xb.shape
        Cout = yb.shape[2]

        lens, sort_idx = lens.sort(descending=True)
        xb, yb = xb[sort_idx], yb[sort_idx]

        h0 = (
            torch.zeros(model.cfg.num_layers, B, model.hidden_size, device=dev),
            torch.zeros(model.cfg.num_layers, B, model.hidden_size, device=dev)
        )

        packed_in         = pack_padded_sequence(xb, lens.cpu(),
                                                 batch_first=True)
        packed_out, _     = model.lstm(packed_in, h0)
        out, _            = pad_packed_sequence(packed_out,
                                                batch_first=True,
                                                total_length=Tmax)  # (B,Tmax,H)

        wl, mu, ls = model.mdn(out.reshape(-1, model.hidden_size))
        wl = wl.reshape(B, Tmax, -1)
        mu = mu.reshape(B, Tmax, model.cfg.num_gaussians, Cout)
        ls = ls.reshape_as(mu)

        mask    = (torch.arange(Tmax, device=dev)[None, :] < lens[:, None])
        mask_f  = mask.reshape(-1)

        target  = yb.reshape(-1, Cout)[mask_f]
        out_kw  = {
            "weight_logits": wl.reshape(-1, model.cfg.num_gaussians)[mask_f],
            "means":         mu.reshape(-1, model.cfg.num_gaussians, Cout)[mask_f],
            "log_stds":      ls.reshape(-1, model.cfg.num_gaussians, Cout)[mask_f],
        }
        loss = model.loss(target, **out_kw)

        tot_nll    += loss.item() * lens.sum().item()
        tot_frames += lens.sum().item()

    return tot_nll / tot_frames
