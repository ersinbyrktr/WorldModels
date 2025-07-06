import torch


def evaluate(model, loader, beta):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for x in loader:
            x = x.to(next(model.parameters()).device)
            xr, mu, lv = model(x)
            rc, kl = model.loss_terms(x, xr, mu, lv)
            tot += rc.item() + beta * kl.item()
    return tot / len(loader)
