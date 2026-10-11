"""Visual inspection: three frozen baselines and an adaptive mathematical fusion.

Reuses existing PatchCore, PaDiM and ViT-Patch checkpoints. Fusion is fitted on
held-out normal images plus synthetic defects, never real test labels. Baselines
retain the previous project's implementations and preprocessing. This is a
project proposal, not a claim of established superiority or research novelty.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch import nn
from torchvision import transforms as T
from tqdm.auto import tqdm

RESIZE, CROP = 256, 224
BASELINES = ("PatchCore", "PaDiM", "ViT-Patch")
METHODS = BASELINES + ("EqualHybrid", "AdaptiveHybrid")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NORMALIZE = T.Compose([T.ToTensor(), T.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
MASK_TRANSFORM = T.Compose([T.Resize(RESIZE, interpolation=T.InterpolationMode.NEAREST),
                           T.CenterCrop(CROP)])

@dataclass
class Config:
    seed: int = 0
    batch_size: int = 4
    calibration_quantile: float = 0.95
    bank_ratio: float = 0.05
    max_bank: int = 4096
    max_candidates: int = 30000
    projection_dim: int = 128
    query_chunk: int = 256
    bank_chunk: int = 2048
    padim_dim: int = 100

    def __post_init__(self):
        if not 0 < self.calibration_quantile < 1 or not 0 < self.bank_ratio <= 1:
            raise ValueError("Invalid quantile or bank ratio.")
        for key in ("batch_size", "max_bank", "max_candidates", "projection_dim",
                    "query_chunk", "bank_chunk", "padim_dim"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive.")



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def load_category(root, category):
    base = Path(root).resolve() / category
    train = list_images(base / "train" / "good")
    if len(train) < 20:
        raise ValueError(f"Need at least 20 normal images in {base / 'train/good'}.")
    if not (base / "test").is_dir():
        raise FileNotFoundError(f"Missing {base / 'test'}")
    test = []
    for folder in sorted((base / "test").iterdir()):
        if not folder.is_dir():
            continue
        for path in list_images(folder):
            label = int(folder.name != "good")
            mask = base / "ground_truth" / folder.name / f"{path.stem}_mask.png" if label else None
            if mask is not None and not mask.is_file():
                raise FileNotFoundError(f"Missing mask: {mask}")
            test.append((path, label, mask))
    if {label for _, label, _ in test} != {0, 1}:
        raise ValueError("Test set must have normal and defective images.")
    return train, test


def split_normal(paths, seed=0):
    paths = sorted(Path(p) for p in paths)
    if len(paths) < 20 or len(set(paths)) != len(paths):
        raise ValueError("Need at least 20 unique normal images.")
    order = np.random.default_rng(seed).permutation(len(paths))
    n = max(2, round(0.1 * len(paths)))
    return {"fit": [paths[i] for i in order[2*n:]],
            "validation": [paths[i] for i in order[n:2*n]],
            "calibration": [paths[i] for i in order[:n]]}


def crop_view(src):
    if isinstance(src, Image.Image):
        img = src.convert("RGB")
    else:
        with Image.open(src) as opened:
            img = opened.convert("RGB")
    return T.functional.center_crop(T.functional.resize(
        img, RESIZE, interpolation=T.InterpolationMode.BICUBIC), CROP)


def image_tensor(src):
    return NORMALIZE(crop_view(src))


def image_batches(paths, batch_size):
    for start in range(0, len(paths), batch_size):
        yield torch.stack([image_tensor(p) for p in paths[start:start+batch_size]])


def maps_from_patches(patch_scores):
    maps = F.interpolate(patch_scores[:, None], size=(CROP, CROP), mode="bilinear",
                         align_corners=False)[:, 0].detach().cpu().numpy()
    maps = np.stack([gaussian_filter(m, sigma=4) for m in maps])
    return maps.reshape(len(maps), -1).max(1), maps


def calibrate_threshold(scores, quantile=0.95):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all() or not 0 < quantile < 1:
        raise ValueError("Need finite calibration scores and a quantile in (0, 1).")
    return float(np.quantile(scores, quantile, method="higher"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


class Detector:
    name = "base"

    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else get_device()
        self.threshold, self.calibration_scores = None, []
        self.fit_seconds, self.history = 0.0, []

    def predict(self, src):
        scores, maps = self.score_tensor(image_tensor(src).unsqueeze(0))
        return float(scores[0]), maps[0]

    def calibrate(self, paths):
        self.calibration_scores = [self.predict(path)[0] for path in paths]
        self.threshold = calibrate_threshold(self.calibration_scores, self.cfg.calibration_quantile)
        return self.threshold

    def fitted_state(self):
        raise NotImplementedError

    def restore_state(self, state):
        raise NotImplementedError

    def save(self, path, signature):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Fitted states use flat CPU tensors; frozen pretrained backbones are reconstructed.
        state = {key: value.cpu() if torch.is_tensor(value) else value
                 for key, value in self.fitted_state().items()}
        payload = {"name": self.name, "config": asdict(self.cfg), "signature": signature,
                   "state": state, "threshold": self.threshold,
                   "calibration_scores": self.calibration_scores,
                   "fit_seconds": self.fit_seconds, "history": self.history}
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def memory_mb(self):
        """Model parameters/buffers plus fitted tensors, excluding transient activations."""
        tensors = list(self.net.parameters()) + list(self.net.buffers())
        tensors += [v for v in self.fitted_state().values() if torch.is_tensor(v)]
        seen, total = set(), 0
        for tensor in tensors:
            key = (str(tensor.device), tensor.data_ptr())
            if key not in seen:
                seen.add(key)
                total += tensor.numel() * tensor.element_size()
        return total / 1024**2


def make_backbone(name, indices, device):
    net = timm.create_model(name, pretrained=True, features_only=True,
                            out_indices=indices).to(device).eval()
    return net.requires_grad_(False)


@torch.no_grad()
def select_coreset(features, cfg, device, size=None):
    n, dim = features.shape
    size = size or min(cfg.max_bank, max(1, int(n * cfg.bank_ratio)))
    generator = torch.Generator().manual_seed(cfg.seed)
    limit = min(n, max(size, cfg.max_candidates))
    candidates = torch.randperm(n, generator=generator)[:limit]
    if size >= limit:
        return features[candidates].float().to(device)
    projection = torch.randn(dim, cfg.projection_dim, generator=generator) / cfg.projection_dim**0.5
    z = torch.cat([features[candidates[i:i+2048]].float() @ projection
                   for i in range(0, limit, 2048)]).to(device)
    norms = (z * z).sum(1)
    minimum = torch.full((limit,), float("inf"), device=device)
    index = int(torch.randint(limit, (1,), generator=generator).item())
    selected = []
    for _ in tqdm(range(size), desc="Coreset", leave=False):
        selected.append(index)
        distance = (norms + norms[index] - 2 * (z @ z[index])).clamp_min(0)
        minimum = torch.minimum(minimum, distance)
        minimum[index] = -1
        index = int(minimum.argmax().item())
    return features[candidates[torch.tensor(selected)]].float().to(device)


@torch.no_grad()
def nearest_distances(queries, bank, cfg):
    output = []
    for query in queries.split(cfg.query_chunk):
        minimum = torch.full((len(query),), float("inf"), device=query.device)
        for block in bank.split(cfg.bank_chunk):
            minimum = torch.minimum(minimum, torch.cdist(query.float(), block.float()).min(1).values)
        output.append(minimum)
    return torch.cat(output)


class MemoryDetector(Detector):
    @torch.no_grad()
    def fit(self, paths):
        pool = []
        for batch in tqdm(image_batches(paths, self.cfg.batch_size), desc=f"Fit {self.name}"):
            f = self.embed(batch)
            pool.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).half().cpu())
        self.bank = select_coreset(torch.cat(pool), self.cfg, self.device)
        return self

    @torch.no_grad()
    def score_tensor(self, x):
        f = self.embed(x)
        b, c, h, w = f.shape
        q = f.permute(0, 2, 3, 1).reshape(-1, c)
        distances = nearest_distances(q, self.bank, self.cfg).reshape(b, h, w)
        return maps_from_patches(distances)

    def fitted_state(self):
        return {"bank": self.bank}

    def restore_state(self, state):
        self.bank = state["bank"].to(self.device)


class PatchCore(MemoryDetector):
    name = "PatchCore"

    def __init__(self, cfg=None, device=None):
        super().__init__(cfg or Config(), device)
        self.net = make_backbone("wide_resnet50_2", (2, 3), self.device)

    @torch.no_grad()
    def embed(self, x):
        f2, f3 = self.net(x.to(self.device))
        f2, f3 = F.avg_pool2d(f2, 3, 1, 1), F.avg_pool2d(f3, 3, 1, 1)
        return torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                                          align_corners=False)], dim=1)


class ViTPatch(MemoryDetector):
    name = "ViT-Patch"

    def __init__(self, cfg=None, device=None):
        super().__init__(cfg or Config(), device)
        self.net = timm.create_model("vit_small_patch16_224.dino", pretrained=True).to(self.device).eval()
        self.net.requires_grad_(False)
        self.tokens = {}
        for block in (7, 11):
            self.net.blocks[block].register_forward_hook(
                lambda module, inputs, output, block=block: self.tokens.__setitem__(block, output))

    @torch.no_grad()
    def embed(self, x):
        self.net.forward_features(x.to(self.device))
        features = []
        for block in (7, 11):
            tokens = self.tokens[block][:, self.net.num_prefix_tokens:]
            grid = int(round(tokens.shape[1]**0.5))
            f = tokens.transpose(1, 2).reshape(len(tokens), -1, grid, grid)
            features.append(F.avg_pool2d(f, 3, 1, 1))
        self.tokens.clear()
        return torch.cat(features, dim=1)


class PaDiM(Detector):
    name = "PaDiM"

    def __init__(self, cfg=None, device=None):
        super().__init__(cfg or Config(), device)
        self.net = make_backbone("resnet18", (1, 2, 3), self.device)
        channels = sum(self.net.feature_info.channels())
        generator = torch.Generator().manual_seed(self.cfg.seed)
        self.idx = torch.randperm(channels, generator=generator)[:self.cfg.padim_dim].to(self.device)

    @torch.no_grad()
    def embed(self, x):
        features = self.net(x.to(self.device))
        size = features[0].shape[-2:]
        f = torch.cat([F.interpolate(t, size=size, mode="bilinear", align_corners=False)
                       for t in features], dim=1)
        return f[:, self.idx]

    @torch.no_grad()
    def fit(self, paths):
        embeddings = torch.cat([self.embed(b) for b in tqdm(
            image_batches(paths, self.cfg.batch_size), desc="Fit PaDiM")])
        n, c, h, w = embeddings.shape
        e = embeddings.reshape(n, c, h*w)
        self.mean = e.mean(0)
        difference = e - self.mean
        covariance = torch.einsum("ncp,ndp->pcd", difference, difference) / (n - 1)
        covariance += 0.01 * torch.eye(c, device=self.device)
        self.inverse = torch.linalg.inv(covariance)
        return self

    @torch.no_grad()
    def score_tensor(self, x):
        e = self.embed(x)
        b, c, h, w = e.shape
        diff = e.reshape(b, c, h*w) - self.mean
        squared = (torch.einsum("bcp,pcd->bpd", diff, self.inverse) * diff.permute(0, 2, 1)).sum(-1)
        return maps_from_patches(squared.clamp_min(0).sqrt().reshape(b, h, w))

    def fitted_state(self):
        return {"mean": self.mean, "inverse": self.inverse, "idx": self.idx}

    def restore_state(self, state):
        self.mean, self.inverse, self.idx = [state[k].to(self.device) for k in ("mean", "inverse", "idx")]


def synthetic_defect(image, rng, kind=None):
    """224 RGB crop -> synthetic RGB image and known binary defect mask.

    Same-image cut/paste, a colored scratch, or an irregular texture disturbance.
    Masks are supervision for generated defects only, never real test masks.
    """
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < 32:
        raise ValueError("Synthetic generation requires images at least 32 px wide/high.")
    kind = kind or rng.choice(["cutpaste", "scratch", "texture"])
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if kind == "scratch":
        x, y = int(rng.integers(w//8, 7*w//8)), int(rng.integers(h//8, 7*h//8))
        x2, y2 = int(np.clip(x + rng.integers(-w//3, w//3), 0, w-1)), int(np.clip(y + rng.integers(h//8, h//3), 0, h-1))
        draw.line([(x, y), (x2, y2)], fill=255, width=int(rng.integers(2, 7)))
        replacement = np.empty((h, w, 3), dtype=np.uint8)
        replacement[:] = rng.integers(0, 256, 3, dtype=np.uint8)
    elif kind in ("cutpaste", "texture"):
        pw, ph = int(rng.integers(w//10, w//3)), int(rng.integers(h//10, h//3))
        x, y = int(rng.integers(0, w-pw+1)), int(rng.integers(0, h-ph+1))
        if kind == "cutpaste":
            sx, sy = int(rng.integers(0, w-pw+1)), int(rng.integers(0, h-ph+1))
            patch = image.crop((sx, sy, sx+pw, sy+ph)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            replacement = np.asarray(image).copy()
            replacement[y:y+ph, x:x+pw] = np.asarray(patch)
            draw.rectangle((x, y, x+pw-1, y+ph-1), fill=255)
        else:
            vertices = [(x + int(rng.integers(pw)), y + int(rng.integers(ph))) for _ in range(8)]
            draw.polygon(vertices, fill=255)
            noise = rng.integers(0, 256, (max(2, h//16), max(2, w//16), 3), dtype=np.uint8)
            replacement = np.asarray(Image.fromarray(noise).resize((w, h), Image.Resampling.BILINEAR))
    else:
        raise ValueError(f"Unknown synthetic defect: {kind}")
    binary = np.asarray(mask) > 0
    if not binary.any():
        return synthetic_defect(image, rng, "scratch")
    alpha = float(rng.uniform(0.55, 1.0))
    original = np.asarray(image, dtype=np.float32)
    blended = (1 - alpha) * original + alpha * replacement.astype(np.float32)
    result = np.where(binary[..., None], blended, original).clip(0, 255).astype(np.uint8)
    if np.array_equal(result, np.asarray(image)):
        # Cut/paste on a uniform background can otherwise produce a false defect mask.
        if kind != "texture":
            return synthetic_defect(image, rng, "texture")
        raise RuntimeError("Synthetic generator failed to change the image.")
    return Image.fromarray(result), binary.astype(np.uint8)


def classification_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    pred, bad = scores > threshold, labels == 1
    tp, fp, fn = int((pred & bad).sum()), int((pred & ~bad).sum()), int((~pred & bad).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"image_auroc": float(roc_auc_score(labels, scores)), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "false_alarm_rate": fp / max(int((~bad).sum()), 1), "threshold": float(threshold)}


def benchmark(det, paths, repeats=3, max_images=20):
    """Repeated warmed batch-one model timing with inputs already preprocessed.

    Includes CPU→device transfer, model scoring, smoothing and returned maps.
    Excludes image decoding and UI. Report CPU and GPU separately.
    """
    chosen = list(paths)[:max_images]
    if not chosen or repeats < 1:
        raise ValueError("Need images and positive repeats.")
    inputs = [image_tensor(p).unsqueeze(0) for p in chosen]
    for _ in range(5):
        det.score_tensor(inputs[0])
    times = []
    for _ in range(repeats):
        for x in inputs:
            synchronize(det.device)
            start = time.perf_counter()
            det.score_tensor(x)
            synchronize(det.device)
            times.append(1000 * (time.perf_counter() - start))
    return {"benchmark_mean_ms": float(np.mean(times)),
            "benchmark_median_ms": float(np.median(times)),
            "benchmark_p95_ms": float(np.quantile(times, 0.95)),
            "benchmark_runs": len(times), "device": str(det.device)}


def heatmap_overlay(src, amap, threshold, alpha=0.5):
    from matplotlib import colormaps
    base = np.asarray(crop_view(src), dtype=np.float32)
    normalized = np.clip(amap / max(2 * threshold, 1e-8), 0, 1)
    heat = colormaps["inferno"](normalized)[..., :3] * 255
    return np.clip((1-alpha) * base + alpha * heat, 0, 255).astype(np.uint8)


def load_saved_split(root, split_file):
    """Reuse the exact old split, rather than resplitting an existing experiment."""
    root = Path(root).resolve()
    saved = json.loads(Path(split_file).read_text())
    if set(saved) != {"fit", "validation", "calibration"}:
        raise ValueError("Expected fit, validation, and calibration splits.")
    split = {key: [(root / name).resolve() for name in names] for key, names in saved.items()}
    all_paths = [path for paths in split.values() for path in paths]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("Saved splits overlap or contain duplicate paths.")
    if any(not paths for paths in split.values()):
        raise ValueError("All three splits must be nonempty.")
    if any(not path.is_relative_to(root) for path in all_paths):
        raise ValueError("Saved split refers outside the dataset root.")
    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset files in the saved split are missing: {missing[:3]}")
    return split


MODEL_CLASSES = {"PatchCore": PatchCore, "PaDiM": PaDiM, "ViT-Patch": ViTPatch}


def load_detector(path, device=None, expected_signature=None):
    """Load our own baseline checkpoints, including checkpoints from the old notebook."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if expected_signature is not None and state["signature"] != expected_signature:
        raise ValueError("Checkpoint does not match configuration/split/code.")
    if state["name"] not in MODEL_CLASSES:
        raise ValueError("This project now uses only PatchCore, PaDiM and ViT-Patch baselines.")
    valid = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in state["config"].items() if k in valid})
    det = MODEL_CLASSES[state["name"]](cfg, device)
    det.restore_state(state["state"])
    det.threshold = state["threshold"]
    det.calibration_scores = state["calibration_scores"]
    det.fit_seconds = state.get("fit_seconds", 0.0)
    det.checkpoint_signature = state["signature"]
    return det


@dataclass
class FusionConfig:
    ridge: float = 10.0
    temperature: float = 0.10
    maximum_strength: float = 0.50
    maximum_map_strength: float = 1.0
    map_patchcore_prior: float = 1/3
    minimum_selection_gain: float = 0.002
    calibration_quantile: float = 0.95

    def __post_init__(self):
        if self.ridge <= 0 or self.temperature <= 0:
            raise ValueError("Ridge and temperature must be positive.")
        if (not 0 <= self.maximum_strength <= 1 or not 0 <= self.maximum_map_strength <= 1
                or not 0 < self.map_patchcore_prior < 1 or self.minimum_selection_gain < 0):
            raise ValueError("Invalid fusion strength or selection margin.")
        if not 0 < self.calibration_quantile < 1:
            raise ValueError("Invalid calibration quantile.")


class FusionMath:
    """Small mathematical reliability model, fitted separately per category/seed.

    1. Normalize positive scores/maps by held-out normal reference quantiles.
    2. Extract score, spatial concentration and cross-model agreement features.
    3. Two ridge regressions predict image-loss and localization-loss advantages.
    4. Two softmax gates create separate image-score and heatmap weights.
    5. Independently select each gate's strength on source-disjoint synthetic
       validation. Each head may fall back to PatchCore without disabling the other.

    Scores are evidence units, NOT probabilities. Weights indicate allocated
    influence; w_m * score_m is an additive contribution, NOT information gain.
    Image weights combine image scores; heatmap weights combine maps. The image
    score need not equal the heatmap max. Heatmap contributions are additive maps.
    """
    def __init__(self, cfg=None):
        self.cfg = cfg or FusionConfig()
        self.image_strength = self.map_strength = 0.0
        self.threshold = None
        self.selection = []

    @staticmethod
    def _check(scores, maps):
        scores, maps = np.asarray(scores), np.asarray(maps)
        if scores.ndim != 2 or scores.shape[1] != 3 or maps.ndim != 4:
            raise ValueError("Expected scores [N,3] and maps [N,3,H,W].")
        if maps.shape[:2] != scores.shape or min(maps.shape[2:]) < 1:
            raise ValueError("Scores and maps disagree.")
        if not np.isfinite(scores).all() or not np.isfinite(maps).all():
            raise ValueError("Non-finite expert outputs.")
        if (scores < 0).any() or (maps < 0).any():
            raise ValueError("Baseline distances must be nonnegative.")
        return scores.astype(np.float64), maps.astype(np.float32, copy=False)

    def normalize(self, scores, maps):
        scores, maps = self._check(scores, maps)
        return (scores / self.score_scale[None],
                maps / self.map_scale[None, :, None, None])

    @staticmethod
    def features(scores, maps):
        # Spatial descriptors use pooled maps; no labels or masks enter inference.
        n, models, h, w = maps.shape
        if h % 14 == 0 and w % 14 == 0:
            pooled = maps.reshape(n, models, 14, h//14, 14, w//14).mean((3, 5))
        else:
            pooled = maps
        flat = np.log1p(pooled).reshape(n, 3, -1)
        pieces = [np.log1p(scores), flat.mean(2), flat.std(2),
                  np.quantile(flat, 0.99, axis=2), flat.max(2),
                  (pooled > 1).reshape(n, 3, -1).mean(2)]
        centered = flat - flat.mean(2, keepdims=True)
        top = flat >= np.quantile(flat, 0.95, axis=2, keepdims=True)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            numerator = (centered[:, a] * centered[:, b]).sum(1)
            denom = np.sqrt((centered[:, a]**2).sum(1) * (centered[:, b]**2).sum(1))
            corr = numerator / np.maximum(denom, 1e-12)
            intersection = (top[:, a] & top[:, b]).sum(1)
            union = (top[:, a] | top[:, b]).sum(1)
            pieces.extend([corr[:, None], (intersection / np.maximum(union, 1))[:, None]])
        return np.concatenate(pieces, axis=1).astype(np.float64)

    @staticmethod
    def expert_losses(scores, maps, masks):
        # Image loss uses bounded evidence, not a calibrated defect probability.
        labels = masks.reshape(len(masks), -1).any(1)
        image_loss = (scores / (1 + scores) - labels[:, None])**2
        pixel_loss = np.empty_like(image_loss)
        for i, mask in enumerate(masks):
            for j in range(3):
                if mask.any():
                    # Rank-based localization loss is invariant to model distance scale.
                    pixel_loss[i, j] = 1 - roc_auc_score(mask.ravel(), maps[i,j].ravel())
                else:
                    evidence = maps[i,j] / (1 + maps[i,j])
                    pixel_loss[i,j] = (evidence**2).mean()
        return {"image": image_loss, "heatmap": pixel_loss}

    def _adaptive_weights(self, scores, maps, strength, head):
        x = (self.features(scores, maps) - self.feature_mean) / self.feature_std
        coefficients = self.image_coefficients if head == "image" else self.map_coefficients
        advantages = np.column_stack([np.ones(len(x)), x]) @ coefficients
        logits = np.column_stack([np.zeros(len(x)), advantages]) / self.cfg.temperature
        pc_prior = 0.8 if head == "image" else self.cfg.map_patchcore_prior
        logits += np.log(np.array([pc_prior, (1-pc_prior)/2, (1-pc_prior)/2]))
        logits -= logits.max(1, keepdims=True)
        soft = np.exp(logits)
        soft /= soft.sum(1, keepdims=True)
        weights = strength * soft
        weights[:, 0] += 1 - strength
        return weights

    def combine(self, scores, maps, equal=False, image_strength=None, map_strength=None):
        scores, maps = self.normalize(scores, maps)
        if equal:
            image_weights = map_weights = np.full((len(scores), 3), 1/3)
        else:
            image_weights = self._adaptive_weights(scores, maps,
                self.image_strength if image_strength is None else image_strength, "image")
            map_weights = self._adaptive_weights(scores, maps,
                self.map_strength if map_strength is None else map_strength, "heatmap")
        contributions = image_weights * scores
        combined_map = np.einsum('nm,nmhw->nhw', map_weights.astype(np.float32), maps)
        total = contributions.sum(1)
        shares = np.divide(contributions, total[:, None], out=np.zeros_like(contributions),
                           where=total[:, None] > 0)
        map_contributions = map_weights * maps.mean((2,3))
        map_total = map_contributions.sum(1)
        map_shares = np.divide(map_contributions, map_total[:,None], out=np.zeros_like(map_contributions),
                               where=map_total[:,None] > 0)
        return {"scores": total, "maps": combined_map, "weights": image_weights,
                "image_weights": image_weights, "map_weights": map_weights,
                "normalized_scores": scores, "normalized_maps": maps,
                "contributions": contributions, "contribution_shares": shares,
                "map_mean_contributions": map_contributions, "map_contribution_shares": map_shares}

    @staticmethod
    def selection_quality(scores, maps, masks):
        labels = masks.reshape(len(masks), -1).any(1)
        image_auc = float(roc_auc_score(labels, scores))
        # Average per-defect localization AUROC avoids size-weighting large masks.
        aucs = [float(roc_auc_score(mask.ravel(), amap.ravel()))
                for mask, amap in zip(masks, maps) if mask.any() and not mask.all()]
        if not aucs:
            raise ValueError("Selection requires synthetic masks with foreground and background.")
        pixel_auc = float(np.mean(aucs))
        return {"image_auroc": image_auc, "mean_defect_pixel_auroc": pixel_auc,
                "objective": 0.5 * (image_auc + pixel_auc)}

    def fit(self, scores, maps, masks, phases, source_ids):
        scores, maps = self._check(scores, maps)
        masks = np.asarray(masks, dtype=bool)
        phases, source_ids = np.asarray(phases), np.asarray(source_ids)
        if (masks.shape != (len(scores), *maps.shape[2:]) or phases.shape != (len(scores),)
                or source_ids.shape != (len(scores),)):
            raise ValueError("Invalid development metadata.")
        if masks.reshape(len(masks), -1).all(1).any():
            raise ValueError("Synthetic masks must leave some normal background.")
        fit = phases == "gate_fit"
        select = phases == "gate_select"
        if not fit.any() or not select.any() or not np.all(fit | select):
            raise ValueError("Need gate_fit and gate_select samples.")
        if set(source_ids[fit]) & set(source_ids[select]):
            raise ValueError("A source image leaks across fusion fit and selection.")
        normal = fit & ~masks.reshape(len(masks), -1).any(1)
        if normal.sum() < 3:
            raise ValueError("Need at least three gate-fit normal source images.")
        self.score_scale = np.maximum(np.quantile(scores[normal], 0.95, axis=0), 1e-8)
        sampled = maps[normal, :, ::4, ::4].transpose(1, 0, 2, 3).reshape(3, -1)
        self.map_scale = np.maximum(np.quantile(sampled, 0.99, axis=1), 1e-8).astype(np.float32)
        s, m = self.normalize(scores[fit], maps[fit])
        x = self.features(s, m)
        self.feature_mean = x.mean(0)
        self.feature_std = np.maximum(x.std(0), 1e-3)
        x = np.column_stack([np.ones(len(x)), (x-self.feature_mean)/self.feature_std])
        losses = self.expert_losses(s, m, masks[fit])
        penalty = self.cfg.ridge * np.eye(x.shape[1]); penalty[0, 0] = 1e-8
        for head, attr in (("image", "image_coefficients"), ("heatmap", "map_coefficients")):
            target = losses[head][:, :1] - losses[head][:, 1:]
            setattr(self, attr, np.linalg.solve(x.T @ x + penalty, x.T @ target))
        self.selection = []
        for head, maximum, attribute, metric in (
            ("image", self.cfg.maximum_strength, "image_strength", "image_auroc"),
            ("heatmap", self.cfg.maximum_map_strength, "map_strength", "mean_defect_pixel_auroc")):
            rows = []
            for strength in sorted(set([0.0, maximum/2, maximum])):
                kwargs = {"image_strength": 0.0, "map_strength": 0.0}
                kwargs[attribute] = strength
                result = self.combine(scores[select], maps[select], **kwargs)
                quality = self.selection_quality(result["scores"], result["maps"], masks[select])
                rows.append({"head": head, "strength": strength, "selection_metric": quality[metric], **quality})
            reference = rows[0]["selection_metric"]
            eligible = [r for r in rows if r["strength"] > 0
                        and r["selection_metric"] > reference + self.cfg.minimum_selection_gain]
            chosen = max(eligible, key=lambda r: r["selection_metric"])["strength"] if eligible else 0.0
            setattr(self, attribute, chosen)
            self.selection.extend(rows)
        self.image_mode = "adaptive" if self.image_strength > 0 else "PatchCore_fallback"
        self.map_mode = "adaptive" if self.map_strength > 0 else "PatchCore_fallback"
        self.mode = "dual_adaptive" if max(self.image_strength, self.map_strength) > 0 else "PatchCore_fallback"
        return self

    def calibrate(self, scores, maps):
        result = self.combine(scores, maps)
        self.threshold = calibrate_threshold(result["scores"], self.cfg.calibration_quantile)
        self.equal_threshold = calibrate_threshold(
            self.combine(scores, maps, equal=True)["scores"], self.cfg.calibration_quantile)

    def save(self, path, checkpoints, metadata=None):
        path = Path(path)
        arrays = {key: getattr(self, key).tolist() for key in
                  ("score_scale", "map_scale", "feature_mean", "feature_std", "image_coefficients", "map_coefficients")}
        atomic_json(path, {"format_version": 2, "baseline_order": list(BASELINES),
                          "config": asdict(self.cfg), "arrays": arrays,
                          "image_strength": self.image_strength, "map_strength": self.map_strength,
                          "mode": self.mode, "image_mode": self.image_mode, "map_mode": self.map_mode,
                          "threshold": self.threshold, "equal_threshold": self.equal_threshold,
                          "selection": self.selection,
                          "checkpoints": {name: os.path.relpath(Path(cp).resolve(), path.parent.resolve())
                                          for name, cp in checkpoints.items()},
                          "metadata": metadata or {}})

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["format_version"] not in (1, 2) or tuple(data["baseline_order"]) != BASELINES:
            raise ValueError("Unsupported fusion file.")
        if data["format_version"] == 1:
            # Preserve inference from the previous single-gate hybrid exactly.
            data["config"]["map_patchcore_prior"] = 0.8
            data["arrays"]["image_coefficients"] = data["arrays"].pop("coefficients")
            data["arrays"]["map_coefficients"] = data["arrays"]["image_coefficients"]
            data["image_strength"] = data["map_strength"] = data["strength"]
            data["image_mode"] = data["map_mode"] = data["mode"]
        obj = cls(FusionConfig(**data["config"]))
        for key, value in data["arrays"].items():
            setattr(obj, key, np.array(value, dtype=np.float32 if key == "map_scale" else np.float64))
        for key in ("image_strength", "map_strength", "image_mode", "map_mode",
                    "mode", "threshold", "equal_threshold", "selection"):
            setattr(obj, key, data[key])
        return obj, data


class AdaptiveHybrid:
    """Live inference API for the future Streamlit app and honest latency measurement."""
    def __init__(self, experts, fusion, equal=False):
        if tuple(experts) != BASELINES:
            raise ValueError("Experts must be ordered PatchCore, PaDiM, ViT-Patch.")
        self.experts, self.fusion, self.equal = experts, fusion, equal
        self.name = "EqualHybrid" if equal else "AdaptiveHybrid"
        self.device = next(iter(experts.values())).device
        if any(expert.device != self.device for expert in experts.values()):
            raise ValueError("All experts must use the same device.")
        self.threshold = fusion.equal_threshold if equal else fusion.threshold
        self.net = nn.ModuleList([expert.net for expert in experts.values()])

    def details_tensor(self, x):
        outputs = [expert.score_tensor(x) for expert in self.experts.values()]
        scores = np.stack([out[0] for out in outputs], axis=1)
        maps = np.stack([out[1] for out in outputs], axis=1)
        result = self.fusion.combine(scores, maps, equal=self.equal)
        result.update(raw_scores=scores, raw_maps=maps)
        return result

    def score_tensor(self, x):
        result = self.details_tensor(x)
        return result["scores"], result["maps"]

    def predict(self, src):
        scores, maps = self.score_tensor(image_tensor(src).unsqueeze(0))
        return float(scores[0]), maps[0]

    def predict_details(self, src):
        result = self.details_tensor(image_tensor(src).unsqueeze(0))
        return {"hybrid_score": float(result["scores"][0]), "hybrid_heatmap": result["maps"][0],
                "threshold": self.threshold, "anomaly": bool(result["scores"][0] > self.threshold),
                "mode": self.fusion.mode, "image_mode": self.fusion.image_mode,
                "heatmap_mode": self.fusion.map_mode, "models": {
                    name: {"raw_score": float(result["raw_scores"][0, i]),
                           "heatmap": result["raw_maps"][0, i],
                           "normalized_heatmap": result["normalized_maps"][0, i],
                           "normalized_score": float(result["normalized_scores"][0, i]),
                           "weight": float(result["image_weights"][0, i]),
                           "image_weight": float(result["image_weights"][0, i]),
                           "heatmap_weight": float(result["map_weights"][0, i]),
                           "weighted_heatmap": (result["map_weights"][0,i] * result["normalized_maps"][0,i]),
                           "heatmap_mean_contribution": float(result["map_mean_contributions"][0,i]),
                           "heatmap_contribution_share": float(result["map_contribution_shares"][0,i]),
                           "weighted_contribution": float(result["contributions"][0, i]),
                           "contribution_share": float(result["contribution_shares"][0, i])}
                    for i, name in enumerate(BASELINES)}}

    def memory_mb(self):
        return sum(expert.memory_mb() for expert in self.experts.values()) + sum(
            getattr(self.fusion, k).nbytes for k in
            ("score_scale", "map_scale", "feature_mean", "feature_std", "image_coefficients", "map_coefficients")) / 1024**2

    @classmethod
    def load(cls, path, device=None, equal=False):
        fusion, data = FusionMath.load(path)
        experts = {name: load_detector(Path(path).parent / data["checkpoints"][name], device)
                   for name in BASELINES}
        return cls(experts, fusion, equal)


def prediction_code_signature():
    """Hash only expert inference/preprocessing code, so gate edits reuse predictions."""
    names = {"Config", "Detector", "MemoryDetector", "PatchCore", "PaDiM", "ViTPatch",
             "get_device", "synchronize", "make_backbone", "nearest_distances",
             "maps_from_patches", "crop_view", "image_tensor", "load_detector"}
    constants = {"RESIZE", "CROP", "NORMALIZE", "MASK_TRANSFORM"}
    nodes = []
    for node in ast.parse(Path(__file__).read_text(encoding="utf-8")).body:
        if getattr(node, "name", None) in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            assigned = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            if assigned & constants:
                nodes.append(node)
    payload = ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()
