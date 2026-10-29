"""Frozen-backbone features for the additional benchmarks.

Writes the same ``.npz`` layout as :func:`redqueen.data.extract` (``emb``,
``spec``, ``y``, ``split``, ``files``) so that :func:`redqueen.data.prepare`
and everything after it are unchanged, and uses the same backbone and
preprocessing helpers (:func:`redqueen.data.backbone`,
:func:`redqueen.data.to_imagenet`). ``spec`` holds per-channel mean and std of
the RGB image (6-d) for these RGB/grey datasets.

Backbones follow the study each dataset is compared with:

- ``so2sat`` (So2Sat LCZ42 RGB, 17 classes; Fan et al. 2023/2024) and
  ``ucmerced`` (UC Merced, 21 classes; Otgonbaatar et al. 2023): ImageNet
  ResNet50, as for EuroSAT.
- ``breastmnist`` (2 classes; Hashmi et al. 2026) and ``cifar10``
  (10 classes; Fulton et al. 2026): ImageNet ResNet18.

Splits: official train/val/test where they exist (So2Sat, BreastMNIST);
CIFAR-10 keeps its test set and takes a stratified 10 % of train as val;
UC Merced has no official split and gets a stratified 70/10/20 split
(seed 0).
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .data import backbone, to_imagenet

ARCH = {"so2sat": "resnet50", "ucmerced": "resnet50", "breastmnist": "resnet18",
        "cifar10": "resnet18"}


def default_out(name: str) -> str:
    return f"features/{name}_{ARCH[name]}.npz"


def _split_labels(n_tr, n_va, n_te):
    return np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * n_te)


def load_so2sat(root: Path):
    parts = [np.load(root / f"{s}.npz") for s in ("train", "valid", "test")]
    x = np.concatenate([p["images"] for p in parts])
    y = np.concatenate([p["labels"] for p in parts])
    return x, y, _split_labels(*(len(p["labels"]) for p in parts))


def load_cifar10(root: Path):
    def batch(name):
        with open(root / "cifar-10-batches-py" / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"])
    tr = [batch(f"data_batch_{i}") for i in range(1, 6)]
    xtr, ytr = np.concatenate([b[0] for b in tr]), np.concatenate([b[1] for b in tr])
    xte, yte = batch("test_batch")
    i_tr, i_va = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=0)
    x = np.concatenate([xtr[i_tr], xtr[i_va], xte])
    y = np.concatenate([ytr[i_tr], ytr[i_va], yte])
    return x, y, _split_labels(len(i_tr), len(i_va), len(yte))


def load_breastmnist(root: Path):
    z = np.load(root / "breastmnist_224.npz")
    xs = [z[f"{s}_images"] for s in ("train", "val", "test")]
    ys = [z[f"{s}_labels"].reshape(-1) for s in ("train", "val", "test")]
    x = np.repeat(np.concatenate(xs)[..., None], 3, axis=-1)   # grey -> 3 channels
    return x, np.concatenate(ys), _split_labels(*(len(v) for v in ys))


def load_ucmerced(root: Path, size: int = 224):
    """Images are resized once, with PIL's antialiased bilinear filter, to the
    backbone's 224 px (the later 224 -> 224 interpolation is then a no-op)."""
    import pandas as pd
    from PIL import Image
    df = pd.read_parquet(root / "ucmerced.parquet")
    img_col = next(c for c in df.columns if "image" in c.lower())
    lab_col = next(c for c in df.columns if "label" in c.lower())
    imgs = []
    for v in df[img_col]:
        if not isinstance(v, dict) or v.get("bytes") is None:
            raise ValueError("UC Merced parquet: expected inline image bytes in every row")
        im = Image.open(io.BytesIO(v["bytes"])).convert("RGB")
        imgs.append(np.asarray(im.resize((size, size), Image.BILINEAR)))
    x, y = np.stack(imgs), df[lab_col].to_numpy()
    idx = np.arange(len(y))
    i_tr, i_rest = train_test_split(idx, test_size=0.3, stratify=y, random_state=0)
    i_va, i_te = train_test_split(i_rest, test_size=2 / 3, stratify=y[i_rest], random_state=0)
    order = np.concatenate([i_tr, i_va, i_te])
    return x[order], y[order], _split_labels(len(i_tr), len(i_va), len(i_te))


LOADERS = {"so2sat": load_so2sat, "cifar10": load_cifar10, "breastmnist": load_breastmnist,
           "ucmerced": load_ucmerced}


def check_labels(y: np.ndarray, name: str) -> np.ndarray:
    """Labels must be 1-D integers 0..C-1 with every class present."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name}: labels must be 1-D, got shape {y.shape} (one-hot?)")
    y = y.astype(int)
    if y.min() != 0 or len(np.unique(y)) != y.max() + 1:
        raise ValueError(f"{name}: labels must be 0..C-1 with no gaps, got {np.unique(y)}")
    return y


@torch.no_grad()
def embed(x: np.ndarray, arch: str, device: str = "cuda", batch: int = 256):
    """(embedding, 6-d RGB mean/std) of uint8 HxWx3 images, batch by batch."""
    net = backbone(arch, device)
    emb, spec = [], []
    for i in range(0, len(x), batch):
        t = torch.from_numpy(x[i:i + batch]).permute(0, 3, 1, 2).float().div(255.0)
        spec.append(torch.cat([t.mean((2, 3)), t.std((2, 3))], 1).numpy())
        emb.append(net(to_imagenet(t, device)).float().cpu().numpy())
    return np.concatenate(emb), np.concatenate(spec)


def extract(name: str, root: str, out: str | None = None, device: str = "cuda") -> None:
    out = out or default_out(name)
    x, y, split = LOADERS[name](Path(root))
    if x.dtype != np.uint8 or x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"{name}: expected uint8 NxHxWx3 images, got {x.dtype} {x.shape}")
    y = check_labels(y, name)
    emb, spec = embed(x, ARCH[name], device)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emb=emb, spec=spec, y=y, split=split,
                        files=np.array([f"{name}_{i}" for i in range(len(y))]))
    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    print(f"saved {out}: {len(y)} images, {y.max() + 1} classes, {counts}, backbone {ARCH[name]}")
