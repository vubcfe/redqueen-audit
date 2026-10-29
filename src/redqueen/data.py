"""EuroSAT-MS features and label-budget splits.

``extract`` runs once: a frozen ImageNet ResNet50 on RGB (B4, B3, B2),
upsampled to 224, gives a 2048-d embedding; per-band mean and std of the 13
bands give a 26-d spectral view. Both are cached in one ``.npz``.

``prepare`` builds one (budget, seed) split. PCA and standardisation are fitted
on unlabeled train images only; the test split is only transformed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

SPLITS = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
RGB = (3, 2, 1)  # B4, B3, B2 (0-based band index)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _Tiles(torch.utils.data.Dataset):
    def __init__(self, root: Path, files: list[str]):
        self.root, self.files = root, files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import rasterio
        with rasterio.open(self.root / self.files[i]) as src:
            return torch.from_numpy(src.read().astype(np.float32))


def backbone(arch: str, device: str):
    """Frozen ImageNet ResNet with the classifier removed (2048-d or 512-d)."""
    import torchvision
    weights = {"resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
               "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V1}[arch]
    net = getattr(torchvision.models, arch)(weights=weights)
    net.fc = torch.nn.Identity()
    return net.eval().to(device)


def to_imagenet(rgb01: torch.Tensor, device: str) -> torch.Tensor:
    """RGB in [0, 1], shape (B, 3, H, W) -> 224 x 224, ImageNet-normalised.
    ``antialias`` only acts when downsampling, so upsampled inputs (EuroSAT
    64 px, So2Sat/CIFAR 32 px) are unchanged by it."""
    x = torch.nn.functional.interpolate(rgb01, size=224, mode="bilinear", align_corners=False,
                                        antialias=True)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).to(device)


def extract(root: str, out: str, device: str = "cuda", batch: int = 256, workers: int = 8) -> None:
    root = Path(root)
    frames = []
    for split, csv in SPLITS.items():
        df = pd.read_csv(root / csv)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    net = backbone("resnet50", device)
    loader = torch.utils.data.DataLoader(_Tiles(root, df["Filename"].tolist()), batch_size=batch,
                                         num_workers=workers)
    emb, spec = [], []
    with torch.no_grad():
        for i, x in enumerate(loader):
            spec.append(torch.cat([x.mean((2, 3)), x.std((2, 3))], 1).numpy())
            rgb = (x[:, list(RGB)] / 3000.0).clamp(0, 1)
            emb.append(net(to_imagenet(rgb, device)).float().cpu().numpy())
            if i % 20 == 0:
                print(f"  batch {i}/{len(loader)}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emb=np.concatenate(emb), spec=np.concatenate(spec),
                        y=df["Label"].to_numpy(), split=df["split"].to_numpy(),
                        files=df["Filename"].to_numpy())
    print(f"saved {out}: {len(df)} tiles")


def _view(z, view: str, fit_idx: np.ndarray) -> np.ndarray:
    """Feature view; any statistics are taken on ``fit_idx`` (unlabeled train)."""
    if view == "resnet":
        return z["emb"]
    spec = np.log1p(z["spec"])
    if view == "spectral":
        return spec
    if view != "both":
        raise ValueError(f"unknown view {view!r}")
    # standardise each block so neither dominates the PCA
    std = lambda a: (a - a[fit_idx].mean(0)) / (a[fit_idx].std(0) + 1e-6)  # noqa: E731
    e, s = std(z["emb"]), std(spec)
    return np.concatenate([e, s * np.sqrt(e.shape[1] / s.shape[1])], 1)


def prepare(features: str, budget: float, seed: int, dim: int, view: str = "resnet",
            n_pool_fit: int = 3000, n_pool_eval: int = 2000, n_pool_audit: int = 4000,
            val_frac: float = 0.2, device: str = "cuda", keep_raw: bool = False) -> dict:
    """One (budget, seed) split. Unlabeled images are split into three
    disjoint pools: ``pool_fit``/``pool_eval`` for the search parasites and
    ``pool_audit`` for the held-out auditors only."""
    z = np.load(features, allow_pickle=True)
    y, split = z["y"].astype(int), z["split"]
    tr, te = np.where(split == "train")[0], np.where(split == "test")[0]
    unlab_extra = np.where(split == "val")[0]

    lab, _ = train_test_split(tr, train_size=budget, stratify=y[tr], random_state=seed)
    # a class with a single labeled sample cannot be stratified: fall back to a
    # plain random split for the search-train / search-val cut
    counts = np.bincount(y[lab])
    strat = y[lab] if counts[counts > 0].min() >= 2 else None
    lab_tr, lab_val = train_test_split(lab, test_size=val_frac, stratify=strat, random_state=seed)
    unlab = np.setdiff1d(np.concatenate([tr, unlab_extra]), lab)
    rng = np.random.default_rng(seed)
    unlab = rng.permutation(unlab)
    # small datasets: shrink the three pools in proportion (recorded in sizes)
    want = n_pool_fit + n_pool_eval + n_pool_audit
    if want > len(unlab):
        s = len(unlab) / want
        n_pool_fit, n_pool_eval = int(n_pool_fit * s), int(n_pool_eval * s)
        n_pool_audit = len(unlab) - n_pool_fit - n_pool_eval
    a, b = n_pool_fit, n_pool_fit + n_pool_eval
    pool_fit, pool_eval, pool_audit = unlab[:a], unlab[a:b], unlab[b:b + n_pool_audit]

    # unsupervised preprocessing on unlabeled train images only
    fit_idx = np.setdiff1d(tr, lab)
    X = _view(z, view, fit_idx)
    pca = PCA(n_components=dim, random_state=seed).fit(X[fit_idx])
    P = pca.transform(X)
    mu, sd = P[fit_idx].mean(0), P[fit_idx].std(0) + 1e-8
    P = (P - mu) / sd

    t = lambda idx: torch.tensor(P[idx], dtype=torch.float32, device=device)  # noqa: E731
    ty = lambda idx: torch.tensor(y[idx], dtype=torch.long, device=device)  # noqa: E731
    out = {"X_tr": t(lab_tr), "y_tr": ty(lab_tr), "X_val": t(lab_val), "y_val": ty(lab_val),
           "X_test": t(te), "y_test": ty(te), "X_pool_fit": t(pool_fit),
           "X_pool_eval": t(pool_eval), "X_pool_audit": t(pool_audit),
           "n_classes": int(y.max() + 1),
           "sizes": {"labeled": len(lab), "search_train": len(lab_tr), "search_val": len(lab_val),
                     "pool_fit": len(pool_fit), "pool_eval": len(pool_eval),
                     "pool_audit": len(pool_audit), "test": len(te)},
           "pca_explained": float(pca.explained_variance_ratio_.sum())}
    if keep_raw:
        out["raw"] = {"tr": X[lab_tr], "val": X[lab_val], "test": X[te]}
    return out
