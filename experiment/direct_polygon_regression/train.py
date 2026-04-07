import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment.direct_polygon_regression.radial_model import RadialPolygonNet


def load_split(path, device):
    d = np.load(path)
    x = torch.from_numpy(d["patches"]).unsqueeze(1).float().to(device)
    y = torch.from_numpy(d["radial_labels"]).float().to(device)
    return x, y


def train(k_dirs: int = 32, epochs: int = 30, batch_size: int = 256, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    split_dir = os.path.join(ROOT, "data", "iris-dataset", "splits")
    tr_path = os.path.join(split_dir, f"train_radial_k{k_dirs}.npz")
    va_path = os.path.join(split_dir, f"val_radial_k{k_dirs}.npz")

    if not os.path.isfile(tr_path) or not os.path.isfile(va_path):
        raise FileNotFoundError(
            f"radial label npz not found. please run prepare_labels.py first. missing: {tr_path} or {va_path}"
        )

    x_tr, y_tr = load_split(tr_path, device)
    x_va, y_va = load_split(va_path, device)

    dl_tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(TensorDataset(x_va, y_va), batch_size=batch_size, shuffle=False)

    model = RadialPolygonNet(k_dirs=k_dirs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best = float("inf")
    save_path = os.path.join(ROOT, "models", f"direct_polygon_regression_k{k_dirs}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_sum = 0
        pbar_tr = tqdm(dl_tr, desc=f"Epoch {ep:03d}/{epochs} [Train]", leave=False)
        for xb, yb in pbar_tr:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)

            loss_main = F.smooth_l1_loss(pred, yb)
            # Encourage smooth radii profile to reduce jagged polygons.
            loss_smooth = (pred[:, 1:] - pred[:, :-1]).abs().mean()
            loss = loss_main + 0.05 * loss_smooth

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            bs = xb.size(0)
            loss_sum += loss.item() * bs
            n_sum += bs
            pbar_tr.set_postfix(loss=f"{loss.item():.5f}")

        tr_loss = loss_sum / max(n_sum, 1)

        model.eval()
        va_sum = 0.0
        va_n = 0
        with torch.no_grad():
            pbar_va = tqdm(dl_va, desc=f"Epoch {ep:03d}/{epochs} [Val]", leave=False)
            for xb, yb in pbar_va:
                pred = model(xb)
                loss = F.smooth_l1_loss(pred, yb)
                bs = xb.size(0)
                va_sum += loss.item() * bs
                va_n += bs
                pbar_va.set_postfix(loss=f"{loss.item():.5f}")
        va_loss = va_sum / max(va_n, 1)

        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), save_path)

        print(f"epoch {ep:03d} | train_loss={tr_loss:.6f} val_loss={va_loss:.6f} best={best:.6f}")

    print("saved best model to:", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_dirs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(k_dirs=args.k_dirs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
