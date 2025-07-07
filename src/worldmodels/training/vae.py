import os

import torch
from tqdm import tqdm

from src.worldmodels.evaluation.vae import evaluate
from src.worldmodels.utils.utils import save_grid


def train(model, train_loader, test_loader, *, epochs: int, lr: float,
          beta: float, kl_on: bool, warm: int, outdir: str, model_path: str = None,
          save_freq: int = 10):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    os.makedirs(outdir, exist_ok=True)

    # Set default model path if not provided
    if model_path is None:
        model_path = os.path.join(outdir, "model_checkpoints")
    os.makedirs(model_path, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=(dev.type == "cuda"))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    fixed = next(iter(test_loader))[:64].to(dev)
    real_row = fixed[:8]

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = run_rc = run_kl = 0.0
        beta_t = beta * min(1.0, ep / warm)

        pbar = tqdm(enumerate(train_loader, 1),
                    total=len(train_loader),
                    desc=f"Epoch {ep}/{epochs}")

        for idx, x in pbar:
            x = x.to(dev, non_blocking=True)

            with torch.amp.autocast(device_type=dev.type,
                                    enabled=dev.type == "cuda",
                                    dtype=torch.float16):
                xr, mu, lv = model(x)
                rc, kl = model.loss_terms(x, xr, mu, lv)
                loss = rc + (beta_t * kl if kl_on else 0.0)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item()
            run_rc += rc.item()
            run_kl += kl.item()

            pbar.set_postfix(loss=run_loss / idx)

        # validation (total loss only – unchanged)
        val = evaluate(model, test_loader, beta_t)

        # averages for this epoch
        avg_loss = run_loss / len(train_loader)
        avg_rc = run_rc / len(train_loader)
        avg_kl = run_kl / len(train_loader)

        # original print
        print(f"Epoch {ep}: train={avg_loss:.4f} val={val:.4f} β={beta_t:.4f}")
        # extra line with rc & kl
        print(f"           rc={avg_rc:.4f} kl={avg_kl:.4f}")

        # sample & recon grids
        save_grid(model.sample(64), f"{outdir}/sample_ep{ep}.png")
        with torch.no_grad():
            recon_row = model.reconstruct(real_row)  # [8, 3, 64, 64]
            recon_grid = torch.cat([real_row, recon_row], dim=0)  # [16, 3, 64, 64]

        save_grid(recon_grid, f"{outdir}/recon_ep{ep}.png", nrow=8)

        # Save model checkpoint
        if (ep % save_freq == 0 or ep == epochs):
            checkpoint_path = os.path.join(model_path, f"vae_checkpoint_ep{ep}.pt")
            model.save_model(checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

            # Save latest model for easy loading
            latest_path = os.path.join(model_path, "vae_latest.pt")
            model.save_model(latest_path)
