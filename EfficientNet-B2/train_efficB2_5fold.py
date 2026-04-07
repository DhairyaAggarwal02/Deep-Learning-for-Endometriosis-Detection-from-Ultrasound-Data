#!/usr/bin/env python3
import os
import time
import json
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix


# -------------------------
# Reproducibility utilities
# -------------------------
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -------------------------
# Augmentations
# -------------------------
class GaussianNoise(torch.nn.Module):
    def __init__(self, std: float = 0.02, p: float = 0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x: torch.Tensor):
        if self.std <= 0:
            return x
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


def build_transforms(img_size: int = 224, train: bool = True) -> transforms.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), shear=5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            GaussianNoise(std=0.02, p=0.5),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# -------------------------
# MixUp + Label smoothing
# -------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def smooth_labels(y: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing


# -------------------------
# Model
# -------------------------
def build_model(pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        model = efficientnet_b2(weights=weights)
    else:
        model = efficientnet_b2(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def set_trainable(model: nn.Module, phase: str):
    for p in model.parameters():
        p.requires_grad = False

    if phase == "head":
        for p in model.classifier.parameters():
            p.requires_grad = True

    elif phase == "layer4+head":
        # EfficientNet-B2 analogue: unfreeze final feature blocks + classifier
        for p in model.features[-2:].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

    elif phase == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown phase: {phase}")


def make_optimizer(model: nn.Module, phase: str, lr_head: float, lr_backbone: float, weight_decay: float):
    if phase == "head":
        params = [{"params": model.classifier.parameters(), "lr": lr_head}]

    elif phase == "layer4+head":
        params = [
            {"params": model.features[-2:].parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ]

    elif phase == "all":
        params = [{"params": model.parameters(), "lr": lr_backbone}]
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return optim.AdamW(params, weight_decay=weight_decay)


# -------------------------
# Dataset helpers
# -------------------------
def infer_patient_id_from_path(path: str) -> str:
    """
    New structure:
      .../fold_k/class_name/patient_id/image.jpg
      .../test/class_name/patient_id/image.jpg
    Return fold/class/patient to avoid collisions.
    """
    patient = os.path.basename(os.path.dirname(path))
    class_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    split_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    return f"{split_name}/{class_name}/{patient}"


def build_patient_ids_from_samples(samples: List[Tuple[str, int]]) -> List[str]:
    return [infer_patient_id_from_path(p) for p, _ in samples]


def get_samples_and_targets(ds) -> Tuple[List[Tuple[str, int]], List[int]]:
    if isinstance(ds, ConcatDataset):
        samples = []
        targets = []
        for sub_ds in ds.datasets:
            samples.extend(sub_ds.samples)
            targets.extend(sub_ds.targets)
        return samples, targets
    return ds.samples, ds.targets


def aggregate_by_patient(probs: np.ndarray, labels: np.ndarray, patient_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from collections import defaultdict

    prob_bucket = defaultdict(list)
    label_bucket = defaultdict(list)

    for pr, lb, pid in zip(probs, labels, patient_ids):
        prob_bucket[pid].append(float(pr))
        label_bucket[pid].append(int(lb))

    pat_ids = sorted(prob_bucket.keys())
    pat_probs = []
    pat_labels = []

    for pid in pat_ids:
        pat_probs.append(float(np.mean(prob_bucket[pid])))
        pat_labels.append(int(round(np.mean(label_bucket[pid]))))

    return np.array(pat_probs, dtype=np.float32), np.array(pat_labels, dtype=np.int32), pat_ids


def find_class_indices(class_to_idx: Dict[str, int]) -> Tuple[int, int]:
    pos_idx, neg_idx = None, None
    for name, idx in class_to_idx.items():
        lname = name.lower()
        if ("endometriosis" in lname) or ("positive" in lname) or ("endo" in lname):
            pos_idx = idx
        if ("healthy" in lname) or ("negative" in lname) or ("control" in lname) or ("ctrl" in lname):
            neg_idx = idx

    if pos_idx is None or neg_idx is None:
        raise RuntimeError(
            f"Could not infer positive/negative classes from folders: {list(class_to_idx.keys())}. "
            f"Expected something like 'endometriosis' vs 'healthy'."
        )
    return pos_idx, neg_idx


def compute_pos_weight_from_dataset(train_ds) -> float:
    _, targets = get_samples_and_targets(train_ds)
    targets = np.array(targets)
    pos = int((targets == 0).sum())
    neg = int((targets == 1).sum())
    # corrected below by explicit class indices in training function
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def compute_pos_weight_with_pos_idx(train_ds, pos_idx: int) -> float:
    _, targets = get_samples_and_targets(train_ds)
    targets = np.array(targets)
    pos = int((targets == pos_idx).sum())
    neg = int((targets != pos_idx).sum())
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def make_fold_dataset(root_dir: str, fold_names: List[str], transform, expected_classes: Optional[List[str]] = None):
    datasets = []
    canonical_classes = None
    canonical_class_to_idx = None

    for fold_name in fold_names:
        fold_path = os.path.join(root_dir, fold_name)
        ds = ImageFolder(fold_path, transform=transform)
        if canonical_classes is None:
            canonical_classes = ds.classes
            canonical_class_to_idx = ds.class_to_idx
        else:
            if ds.classes != canonical_classes:
                raise RuntimeError(
                    f"Class mismatch in {fold_path}. Expected {canonical_classes}, got {ds.classes}"
                )
        datasets.append(ds)

    if expected_classes is not None and canonical_classes != expected_classes:
        raise RuntimeError(f"Expected classes {expected_classes}, got {canonical_classes} in {fold_names}")

    if len(datasets) == 1:
        ds = datasets[0]
        ds.classes = canonical_classes
        ds.class_to_idx = canonical_class_to_idx
        return ds

    combo = ConcatDataset(datasets)
    combo.classes = canonical_classes
    combo.class_to_idx = canonical_class_to_idx
    combo.samples = [s for d in datasets for s in d.samples]
    combo.targets = [t for d in datasets for t in d.targets]
    return combo


# -------------------------
# Threshold search + metrics
# -------------------------
def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(np.int32)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "thr": float(thr),
        "acc": acc,
        "f1": f1,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray, grid: int = 1001) -> Dict[str, Dict[str, float]]:
    thrs = np.linspace(0.0, 1.0, grid, dtype=np.float32)

    best_f1 = -1.0
    best_f1_thr = 0.5

    best_j = -1e9
    best_j_thr = 0.5

    for t in thrs:
        y_pred = (y_prob >= t).astype(np.int32)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = float(t)

        if j > best_j:
            best_j = j
            best_j_thr = float(t)

    return {
        "best_f1": metrics_at_threshold(y_true, y_prob, best_f1_thr),
        "best_youdenJ": metrics_at_threshold(y_true, y_prob, best_j_thr),
    }


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_idx: int,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []

    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_bin = torch.where(y == pos_idx, torch.ones_like(y), torch.zeros_like(y)).float().unsqueeze(1)

        with autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(y_bin.detach().cpu().numpy())

    probs = np.vstack(all_probs).reshape(-1).astype(np.float32)
    labels = np.vstack(all_labels).reshape(-1).astype(np.int32)
    return probs, labels


def evaluate_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_idx: int,
    patient_ids: Optional[List[str]] = None,
    use_amp: bool = True,
) -> Dict[str, Dict[str, float]]:
    probs, labels = predict_probs(model, loader, device, pos_idx, use_amp=use_amp)

    out = {}
    auc_img = safe_auc(labels, probs)
    out["image"] = {
        "auc": float(auc_img),
        "acc@0.5": float(accuracy_score(labels, (probs >= 0.5).astype(np.int32))),
        "f1@0.5": float(f1_score(labels, (probs >= 0.5).astype(np.int32), zero_division=0)),
        "n": int(len(labels)),
    }

    if patient_ids is not None:
        pat_probs, pat_labels, _ = aggregate_by_patient(probs, labels, patient_ids)
        auc_pat = safe_auc(pat_labels, pat_probs)
        out["patient"] = {
            "auc": float(auc_pat),
            "acc@0.5": float(accuracy_score(pat_labels, (pat_probs >= 0.5).astype(np.int32))),
            "f1@0.5": float(f1_score(pat_labels, (pat_probs >= 0.5).astype(np.int32), zero_division=0)),
            "n_patients": int(len(pat_labels)),
        }

    return out


# -------------------------
# Early stopping
# -------------------------
@dataclass
class EarlyStopper:
    patience: int
    min_delta: float = 1e-4
    best: float = -1e9
    bad_epochs: int = 0

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# -------------------------
# Training
# -------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_idx: int,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    grad_clip_norm: float,
    mixup_alpha: float,
    label_smoothing: float,
) -> float:
    model.train()
    running_loss = 0.0
    n_steps = 0

    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_bin = torch.where(y == pos_idx, torch.ones_like(y), torch.zeros_like(y)).float().unsqueeze(1)

        y_bin = smooth_labels(y_bin, label_smoothing)
        x, y_bin = mixup_batch(x, y_bin, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with autocast(True):
                logits = model(x)
                loss = criterion(logits, y_bin)
            scaler.scale(loss).backward()

            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y_bin)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += float(loss.item())
        n_steps += 1

    return running_loss / max(n_steps, 1)


def summarize_metric_list(metric_dicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in metric_dicts]
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "values": vals,
        }
    return out


def train_one_fold(
    fold_idx: int,
    train_fold_names: List[str],
    val_fold_name: str,
    test_fold_name: str,
    args,
    device: torch.device,
    use_amp: bool,
    root_out_dir: str,
    canonical_classes: Optional[List[str]] = None,
):
    fold_out_dir = os.path.join(root_out_dir, f"cv_fold_{fold_idx}")
    os.makedirs(fold_out_dir, exist_ok=True)

    train_ds = make_fold_dataset(args.data_root, train_fold_names, build_transforms(args.img_size, train=True), expected_classes=canonical_classes)
    train_eval_ds = make_fold_dataset(args.data_root, train_fold_names, build_transforms(args.img_size, train=False), expected_classes=train_ds.classes)
    val_ds = make_fold_dataset(args.data_root, [val_fold_name], build_transforms(args.img_size, train=False), expected_classes=train_ds.classes)
    test_ds = make_fold_dataset(args.data_root, [test_fold_name], build_transforms(args.img_size, train=False), expected_classes=train_ds.classes)

    print(f"\n{'=' * 90}")
    print(f"[INFO] CV FOLD {fold_idx}: train={train_fold_names} | val={val_fold_name} | test={test_fold_name}")
    print(f"[INFO] Classes: {train_ds.classes} (class_to_idx={train_ds.class_to_idx})")

    pos_idx, neg_idx = find_class_indices(train_ds.class_to_idx)

    val_patient_ids = build_patient_ids_from_samples(val_ds.samples)
    test_patient_ids = build_patient_ids_from_samples(test_ds.samples)
    train_eval_patient_ids = build_patient_ids_from_samples(train_eval_ds.samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)

    model = build_model(pretrained=(not args.no_pretrained)).to(device)

    pos_weight_value = compute_pos_weight_with_pos_idx(train_ds, pos_idx)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[INFO][FOLD {fold_idx}] pos_weight (neg/pos) = {pos_weight_value:.4f}")

    best_path = os.path.join(fold_out_dir, "best_by_val_patient_auc.pt")
    last_path = os.path.join(fold_out_dir, "last.pt")
    log_path = os.path.join(fold_out_dir, "metrics.jsonl")
    summary_path = os.path.join(fold_out_dir, "summary.json")

    best_metric = -1e9
    best_epoch = -1
    early = EarlyStopper(patience=args.early_patience, min_delta=args.early_min_delta)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def write_jsonl(obj: dict):
        with open(log_path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    phase = "head"
    set_trainable(model, phase)
    optimizer = make_optimizer(model, phase, lr_head=args.lr_head, lr_backbone=args.lr_backbone,
                               weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, verbose=False
    )

    global_epoch = 0
    print(f"[INFO][FOLD {fold_idx}] Phase 1: head-only for {args.phase1_epochs} epochs")

    for _ in range(args.phase1_epochs):
        global_epoch += 1
        start_t = time.time()

        train_loss = train_one_epoch(
            model, train_loader, device, pos_idx, criterion, optimizer,
            scaler=scaler, use_amp=use_amp, grad_clip_norm=args.grad_clip_norm,
            mixup_alpha=args.mixup_alpha, label_smoothing=args.label_smoothing
        )

        val_metrics = evaluate_all(model, val_loader, device, pos_idx, patient_ids=val_patient_ids, use_amp=use_amp)
        val_probs_img, val_labels_img = predict_probs(model, val_loader, device, pos_idx, use_amp=use_amp)
        val_probs_pat, val_labels_pat, val_pat_ids = aggregate_by_patient(val_probs_img, val_labels_img, val_patient_ids)
        val_thr_report = find_best_thresholds(val_labels_pat, val_probs_pat, grid=1001)

        train_eval_metrics = None
        if args.eval_train:
            train_eval_metrics = evaluate_all(model, train_eval_loader, device, pos_idx, patient_ids=train_eval_patient_ids, use_amp=use_amp)

        monitor = float(val_metrics["patient"]["auc"])
        scheduler.step(monitor)
        lr_groups = [pg["lr"] for pg in optimizer.param_groups]

        record = {
            "cv_fold": fold_idx,
            "epoch": global_epoch,
            "phase": "PH1",
            "train_folds": train_fold_names,
            "val_fold": val_fold_name,
            "train": {"loss": float(train_loss)} | (train_eval_metrics if train_eval_metrics else {}),
            "val": val_metrics,
            "val_thresholds_patient": val_thr_report,
            "lr": lr_groups,
            "time_sec": float(time.time() - start_t),
        }
        write_jsonl(record)

        print(f"[VAL][FOLD {fold_idx}][PH1] epoch={global_epoch} "
              f"val_patient_auc={val_metrics['patient']['auc']:.4f} "
              f"val_image_auc={val_metrics['image']['auc']:.4f} "
              f"train_loss={train_loss:.4f} lr={lr_groups}")

        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": global_epoch}, last_path)

        if monitor > best_metric:
            best_metric = monitor
            best_epoch = global_epoch
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": global_epoch,
                "best_val_patient_auc": best_metric,
                "cv_fold": fold_idx,
                "train_folds": train_fold_names,
                "val_fold": val_fold_name,
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "pos_idx": pos_idx,
                "neg_idx": neg_idx,
            }, best_path)
            print(f"[INFO][FOLD {fold_idx}] New best val PATIENT AUC: {best_metric:.4f}")

        if early.step(monitor):
            print(f"[INFO][FOLD {fold_idx}] Early stopping triggered in Phase 1 at epoch {global_epoch}.")
            break

    remaining_epochs = args.max_epochs - global_epoch
    if remaining_epochs > 0:
        phase = "layer4+head"
        set_trainable(model, phase)
        optimizer = make_optimizer(model, phase, lr_head=1e-4, lr_backbone=args.lr_backbone,
                                   weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience, verbose=False
        )
        early = EarlyStopper(patience=args.early_patience, min_delta=args.early_min_delta)

        print(f"[INFO][FOLD {fold_idx}] Phase 2: unfreeze layer4+head for up to {remaining_epochs} epochs")

        for _ in range(remaining_epochs):
            global_epoch += 1
            start_t = time.time()

            train_loss = train_one_epoch(
                model, train_loader, device, pos_idx, criterion, optimizer,
                scaler=scaler, use_amp=use_amp, grad_clip_norm=args.grad_clip_norm,
                mixup_alpha=args.mixup_alpha, label_smoothing=args.label_smoothing
            )

            val_metrics = evaluate_all(model, val_loader, device, pos_idx, patient_ids=val_patient_ids, use_amp=use_amp)
            val_probs_img, val_labels_img = predict_probs(model, val_loader, device, pos_idx, use_amp=use_amp)
            val_probs_pat, val_labels_pat, val_pat_ids = aggregate_by_patient(val_probs_img, val_labels_img, val_patient_ids)
            val_thr_report = find_best_thresholds(val_labels_pat, val_probs_pat, grid=1001)

            train_eval_metrics = None
            if args.eval_train:
                train_eval_metrics = evaluate_all(model, train_eval_loader, device, pos_idx, patient_ids=train_eval_patient_ids, use_amp=use_amp)

            monitor = float(val_metrics["patient"]["auc"])
            scheduler.step(monitor)
            lr_groups = [pg["lr"] for pg in optimizer.param_groups]

            record = {
                "cv_fold": fold_idx,
                "epoch": global_epoch,
                "phase": "PH2",
                "train_folds": train_fold_names,
                "val_fold": val_fold_name,
                "train": {"loss": float(train_loss)} | (train_eval_metrics if train_eval_metrics else {}),
                "val": val_metrics,
                "val_thresholds_patient": val_thr_report,
                "lr": lr_groups,
                "time_sec": float(time.time() - start_t),
            }
            write_jsonl(record)

            print(f"[VAL][FOLD {fold_idx}][PH2] epoch={global_epoch} "
                  f"val_patient_auc={val_metrics['patient']['auc']:.4f} "
                  f"val_image_auc={val_metrics['image']['auc']:.4f} "
                  f"train_loss={train_loss:.4f} lr={lr_groups}")

            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": global_epoch}, last_path)

            if monitor > best_metric:
                best_metric = monitor
                best_epoch = global_epoch
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": global_epoch,
                    "best_val_patient_auc": best_metric,
                    "cv_fold": fold_idx,
                    "train_folds": train_fold_names,
                    "val_fold": val_fold_name,
                    "classes": train_ds.classes,
                    "class_to_idx": train_ds.class_to_idx,
                    "pos_idx": pos_idx,
                    "neg_idx": neg_idx,
                }, best_path)
                print(f"[INFO][FOLD {fold_idx}] New best val PATIENT AUC: {best_metric:.4f}")

            if early.step(monitor):
                print(f"[INFO][FOLD {fold_idx}] Early stopping triggered in Phase 2 at epoch {global_epoch}.")
                break

    print(f"[INFO][FOLD {fold_idx}] Loading best checkpoint: {best_path}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    val_probs_img, val_labels_img = predict_probs(model, val_loader, device, pos_idx, use_amp=use_amp)
    val_thr_report_img = find_best_thresholds(val_labels_img, val_probs_img, grid=2001)
    val_probs_pat, val_labels_pat, val_pat_ids = aggregate_by_patient(val_probs_img, val_labels_img, val_patient_ids)
    val_thr_report_pat = find_best_thresholds(val_labels_pat, val_probs_pat, grid=2001)

    test_probs_img, test_labels_img = predict_probs(model, test_loader, device, pos_idx, use_amp=use_amp)
    test_probs_pat, test_labels_pat, test_pat_ids = aggregate_by_patient(test_probs_img, test_labels_img, test_patient_ids)

    test_auc_img = safe_auc(test_labels_img, test_probs_img)
    test_auc_pat = safe_auc(test_labels_pat, test_probs_pat)

    img_thr_f1 = val_thr_report_img["best_f1"]["thr"]
    img_thr_j = val_thr_report_img["best_youdenJ"]["thr"]
    pat_thr_f1 = val_thr_report_pat["best_f1"]["thr"]
    pat_thr_j = val_thr_report_pat["best_youdenJ"]["thr"]

    test_img_at_05 = metrics_at_threshold(test_labels_img, test_probs_img, 0.5)
    test_img_at_f1 = metrics_at_threshold(test_labels_img, test_probs_img, img_thr_f1)
    test_img_at_j = metrics_at_threshold(test_labels_img, test_probs_img, img_thr_j)

    test_pat_at_05 = metrics_at_threshold(test_labels_pat, test_probs_pat, 0.5)
    test_pat_at_f1 = metrics_at_threshold(test_labels_pat, test_probs_pat, pat_thr_f1)
    test_pat_at_j = metrics_at_threshold(test_labels_pat, test_probs_pat, pat_thr_j)

    summary = {
        "cv_fold": fold_idx,
        "train_folds": train_fold_names,
        "val_fold": val_fold_name,
        "test_split": test_fold_name,
        "best_val_patient_auc": float(best_metric),
        "best_epoch": int(best_epoch),
        "pos_weight": float(pos_weight_value),
        "classes": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
        "pos_idx": int(pos_idx),
        "neg_idx": int(neg_idx),
        "val_thresholds_image": val_thr_report_img,
        "val_thresholds_patient": val_thr_report_pat,
        "test": {
            "image_auc": float(test_auc_img),
            "patient_auc": float(test_auc_pat),
            "image_metrics@0.5": test_img_at_05,
            "image_metrics@val_best_f1_thr": test_img_at_f1,
            "image_metrics@val_best_youdenJ_thr": test_img_at_j,
            "patient_metrics@0.5": test_pat_at_05,
            "patient_metrics@val_best_f1_thr": test_pat_at_f1,
            "patient_metrics@val_best_youdenJ_thr": test_pat_at_j,
            "n_test_images": int(len(test_labels_img)),
            "n_test_patients": int(len(test_labels_pat)),
        },
        "amp": bool(use_amp),
        "grad_clip_norm": float(args.grad_clip_norm),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        os.path.join(fold_out_dir, "predictions_best_model.npz"),
        val_probs_img=val_probs_img,
        val_labels_img=val_labels_img,
        val_patient_ids=np.array(val_patient_ids),
        val_probs_pat=val_probs_pat,
        val_labels_pat=val_labels_pat,
        val_pat_ids=np.array(val_pat_ids),
        test_probs_img=test_probs_img,
        test_labels_img=test_labels_img,
        test_patient_ids=np.array(test_patient_ids),
        test_probs_pat=test_probs_pat,
        test_labels_pat=test_labels_pat,
        test_pat_ids=np.array(test_pat_ids),
    )

    print(f"[TEST][FOLD {fold_idx}] image_auc={test_auc_img:.4f} | patient_auc={test_auc_pat:.4f}")
    print(f"[INFO][FOLD {fold_idx}] Wrote: {log_path}")
    print(f"[INFO][FOLD {fold_idx}] Wrote: {summary_path}")

    return {
        "fold_idx": fold_idx,
        "fold_out_dir": fold_out_dir,
        "best_path": best_path,
        "summary_path": summary_path,
        "classes": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "val_probs_img": val_probs_img,
        "val_labels_img": val_labels_img,
        "val_patient_ids": np.array(val_patient_ids),
        "val_probs_pat": val_probs_pat,
        "val_labels_pat": val_labels_pat,
        "val_pat_ids": np.array(val_pat_ids),
        "test_probs_img": test_probs_img,
        "test_labels_img": test_labels_img,
        "test_patient_ids": np.array(test_patient_ids),
        "test_probs_pat": test_probs_pat,
        "test_labels_pat": test_labels_pat,
        "test_pat_ids": np.array(test_pat_ids),
        "val_thresholds_image": val_thr_report_img,
        "val_thresholds_patient": val_thr_report_pat,
        "test_summary": summary["test"],
        "best_val_patient_auc": best_metric,
        "best_epoch": best_epoch,
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root containing fold_1..fold_5 and test/")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=260)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--phase1_epochs", type=int, default=15)
    parser.add_argument("--early_patience", type=int, default=10)
    parser.add_argument("--early_min_delta", type=float, default=1e-4)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--lr_factor", type=float, default=0.5)

    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--eval_train", action="store_true")
    parser.add_argument("--fold_prefix", type=str, default="fold_")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--test_dir_name", type=str, default="test")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    print(f"[INFO] Device: {device} | AMP: {use_amp}")

    fold_names = [f"{args.fold_prefix}{i}" for i in range(1, args.n_folds + 1)]
    for fold_name in fold_names + [args.test_dir_name]:
        fold_path = os.path.join(args.data_root, fold_name)
        if not os.path.isdir(fold_path):
            raise FileNotFoundError(f"Missing required directory: {fold_path}")

    probe_ds = ImageFolder(os.path.join(args.data_root, fold_names[0]))
    canonical_classes = probe_ds.classes
    print(f"[INFO] Using dataset classes: {canonical_classes} (class_to_idx={probe_ds.class_to_idx})")

    fold_results = []
    for i, val_fold in enumerate(fold_names, start=1):
        train_folds = [f for f in fold_names if f != val_fold]
        fold_result = train_one_fold(
            fold_idx=i,
            train_fold_names=train_folds,
            val_fold_name=val_fold,
            test_fold_name=args.test_dir_name,
            args=args,
            device=device,
            use_amp=use_amp,
            root_out_dir=args.out_dir,
            canonical_classes=canonical_classes,
        )
        fold_results.append(fold_result)

    # -------------------------------------------------
    # Aggregate all validation predictions across folds
    # -------------------------------------------------
    oof_val_probs_img = np.concatenate([fr["val_probs_img"] for fr in fold_results])
    oof_val_labels_img = np.concatenate([fr["val_labels_img"] for fr in fold_results])
    oof_val_patient_ids = np.concatenate([fr["val_patient_ids"] for fr in fold_results]).tolist()

    oof_val_probs_pat, oof_val_labels_pat, oof_val_pat_ids = aggregate_by_patient(
        oof_val_probs_img, oof_val_labels_img, oof_val_patient_ids
    )

    cv_val_auc_img = safe_auc(oof_val_labels_img, oof_val_probs_img)
    cv_val_auc_pat = safe_auc(oof_val_labels_pat, oof_val_probs_pat)
    cv_val_thr_img = find_best_thresholds(oof_val_labels_img, oof_val_probs_img, grid=2001)
    cv_val_thr_pat = find_best_thresholds(oof_val_labels_pat, oof_val_probs_pat, grid=2001)

    # -------------------------------------------------
    # Ensemble the 5 CV models on the separate test set
    # -------------------------------------------------
    test_labels_img_ref = fold_results[0]["test_labels_img"]
    test_patient_ids_ref = fold_results[0]["test_patient_ids"]
    test_pat_ids_ref = fold_results[0]["test_pat_ids"]
    test_labels_pat_ref = fold_results[0]["test_labels_pat"]

    for fr in fold_results[1:]:
        if not np.array_equal(test_labels_img_ref, fr["test_labels_img"]):
            raise RuntimeError("Test labels/order mismatch across folds. Keep the same test dataset and deterministic ordering.")
        if not np.array_equal(test_patient_ids_ref, fr["test_patient_ids"]):
            raise RuntimeError("Test patient/image ordering mismatch across folds.")
        if not np.array_equal(test_pat_ids_ref, fr["test_pat_ids"]):
            raise RuntimeError("Patient-level test ordering mismatch across folds.")
        if not np.array_equal(test_labels_pat_ref, fr["test_labels_pat"]):
            raise RuntimeError("Patient-level test labels mismatch across folds.")

    ensemble_test_probs_img = np.mean(np.stack([fr["test_probs_img"] for fr in fold_results], axis=0), axis=0)
    ensemble_test_probs_pat = np.mean(np.stack([fr["test_probs_pat"] for fr in fold_results], axis=0), axis=0)

    final_test_auc_img = safe_auc(test_labels_img_ref, ensemble_test_probs_img)
    final_test_auc_pat = safe_auc(test_labels_pat_ref, ensemble_test_probs_pat)

    img_thr_f1 = cv_val_thr_img["best_f1"]["thr"]
    img_thr_j = cv_val_thr_img["best_youdenJ"]["thr"]
    pat_thr_f1 = cv_val_thr_pat["best_f1"]["thr"]
    pat_thr_j = cv_val_thr_pat["best_youdenJ"]["thr"]

    final_test_img_at_05 = metrics_at_threshold(test_labels_img_ref, ensemble_test_probs_img, 0.5)
    final_test_img_at_f1 = metrics_at_threshold(test_labels_img_ref, ensemble_test_probs_img, img_thr_f1)
    final_test_img_at_j = metrics_at_threshold(test_labels_img_ref, ensemble_test_probs_img, img_thr_j)

    final_test_pat_at_05 = metrics_at_threshold(test_labels_pat_ref, ensemble_test_probs_pat, 0.5)
    final_test_pat_at_f1 = metrics_at_threshold(test_labels_pat_ref, ensemble_test_probs_pat, pat_thr_f1)
    final_test_pat_at_j = metrics_at_threshold(test_labels_pat_ref, ensemble_test_probs_pat, pat_thr_j)

    per_fold_test_image_auc = [fr["test_summary"]["image_auc"] for fr in fold_results]
    per_fold_test_patient_auc = [fr["test_summary"]["patient_auc"] for fr in fold_results]
    per_fold_test_img_05 = [fr["test_summary"]["image_metrics@0.5"] for fr in fold_results]
    per_fold_test_pat_05 = [fr["test_summary"]["patient_metrics@0.5"] for fr in fold_results]

    # Per-fold test metrics using thresholds selected from each fold's own validation set
    per_fold_test_img_f1 = [fr["test_summary"]["image_metrics@val_best_f1_thr"] for fr in fold_results]
    per_fold_test_img_j = [fr["test_summary"]["image_metrics@val_best_youdenJ_thr"] for fr in fold_results]
    per_fold_test_pat_f1 = [fr["test_summary"]["patient_metrics@val_best_f1_thr"] for fr in fold_results]
    per_fold_test_pat_j = [fr["test_summary"]["patient_metrics@val_best_youdenJ_thr"] for fr in fold_results]

    cv_summary = {
        "dataset_layout": {
            "data_root": args.data_root,
            "cv_folds": fold_names,
            "test_dir": args.test_dir_name,
            "train_each_run_uses": f"{args.n_folds - 1} folds",
            "val_each_run_uses": "1 fold",
        },
        "classes": canonical_classes,
        "cv_fold_summaries": [
            {
                "cv_fold": fr["fold_idx"],
                "best_val_patient_auc": float(fr["best_val_patient_auc"]),
                "best_epoch": int(fr["best_epoch"]),
                "summary_path": fr["summary_path"],
                "best_checkpoint": fr["best_path"],
                "test_image_auc": float(fr["test_summary"]["image_auc"]),
                "test_patient_auc": float(fr["test_summary"]["patient_auc"]),
            }
            for fr in fold_results
        ],
        "cross_validation_validation_pool": {
            "image_auc": float(cv_val_auc_img),
            "patient_auc": float(cv_val_auc_pat),
            "image_metrics@0.5": metrics_at_threshold(oof_val_labels_img, oof_val_probs_img, 0.5),
            "patient_metrics@0.5": metrics_at_threshold(oof_val_labels_pat, oof_val_probs_pat, 0.5),
            "val_thresholds_image": cv_val_thr_img,
            "val_thresholds_patient": cv_val_thr_pat,
            "n_val_images_total": int(len(oof_val_labels_img)),
            "n_val_patients_total": int(len(oof_val_labels_pat)),
        },
        "test_per_fold_metrics": {
            "image_auc": {
                "mean": float(np.mean(per_fold_test_image_auc)),
                "std": float(np.std(per_fold_test_image_auc, ddof=0)),
                "values": per_fold_test_image_auc,
            },
            "patient_auc": {
                "mean": float(np.mean(per_fold_test_patient_auc)),
                "std": float(np.std(per_fold_test_patient_auc, ddof=0)),
                "values": per_fold_test_patient_auc,
            },
            "image_metrics@0.5": summarize_metric_list(per_fold_test_img_05),
            "patient_metrics@0.5": summarize_metric_list(per_fold_test_pat_05),
            "image_metrics@val_best_f1_thr": summarize_metric_list(per_fold_test_img_f1),
            "image_metrics@val_best_youdenJ_thr": summarize_metric_list(per_fold_test_img_j),
            "patient_metrics@val_best_f1_thr": summarize_metric_list(per_fold_test_pat_f1),
            "patient_metrics@val_best_youdenJ_thr": summarize_metric_list(per_fold_test_pat_j),
        },
        "final_test_ensemble": {
            "description": "Average of probabilities from the 5 best CV models, evaluated on the completely separate test set.",
            "image_auc": float(final_test_auc_img),
            "patient_auc": float(final_test_auc_pat),
            "image_metrics@0.5": final_test_img_at_05,
            "image_metrics@cv_best_f1_thr": final_test_img_at_f1,
            "image_metrics@cv_best_youdenJ_thr": final_test_img_at_j,
            "patient_metrics@0.5": final_test_pat_at_05,
            "patient_metrics@cv_best_f1_thr": final_test_pat_at_f1,
            "patient_metrics@cv_best_youdenJ_thr": final_test_pat_at_j,
            "n_test_images": int(len(test_labels_img_ref)),
            "n_test_patients": int(len(test_labels_pat_ref)),
        },
        "amp": bool(use_amp),
        "grad_clip_norm": float(args.grad_clip_norm),
    }

    summary_path = os.path.join(args.out_dir, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(cv_summary, f, indent=2)

    np.savez_compressed(
        os.path.join(args.out_dir, "cv_aggregated_predictions.npz"),
        oof_val_probs_img=oof_val_probs_img,
        oof_val_labels_img=oof_val_labels_img,
        oof_val_patient_ids=np.array(oof_val_patient_ids),
        oof_val_probs_pat=oof_val_probs_pat,
        oof_val_labels_pat=oof_val_labels_pat,
        oof_val_pat_ids=np.array(oof_val_pat_ids),
        ensemble_test_probs_img=ensemble_test_probs_img,
        ensemble_test_labels_img=test_labels_img_ref,
        ensemble_test_patient_ids=test_patient_ids_ref,
        ensemble_test_probs_pat=ensemble_test_probs_pat,
        ensemble_test_labels_pat=test_labels_pat_ref,
        ensemble_test_pat_ids=test_pat_ids_ref,
    )

    print("\n" + "=" * 90)
    print("[FINAL CV SUMMARY]")
    print(f"[OOF VAL] image_auc={cv_val_auc_img:.4f} | patient_auc={cv_val_auc_pat:.4f}")
    print(f"[TEST ENSEMBLE] image_auc={final_test_auc_img:.4f} | patient_auc={final_test_auc_pat:.4f}")
    print(
        f"[TEST ENSEMBLE][PATIENT @0.5] "
        f"acc={final_test_pat_at_05['acc']:.4f} "
        f"sens={final_test_pat_at_05['sensitivity']:.4f} "
        f"spec={final_test_pat_at_05['specificity']:.4f} "
        f"f1={final_test_pat_at_05['f1']:.4f}"
    )
    img_j_summary = summarize_metric_list(per_fold_test_img_j)
    pat_j_summary = summarize_metric_list(per_fold_test_pat_j)

    print(
        f"[TEST 5-FOLD MEAN][IMAGE @ val_best_youdenJ_thr] "
        f"acc={img_j_summary['acc']['mean']:.4f} "
        f"sens={img_j_summary['sensitivity']['mean']:.4f} "
        f"spec={img_j_summary['specificity']['mean']:.4f} "
        f"f1={img_j_summary['f1']['mean']:.4f}"
    )
    print(
        f"[TEST 5-FOLD MEAN][PATIENT @ val_best_youdenJ_thr] "
        f"acc={pat_j_summary['acc']['mean']:.4f} "
        f"sens={pat_j_summary['sensitivity']['mean']:.4f} "
        f"spec={pat_j_summary['specificity']['mean']:.4f} "
        f"f1={pat_j_summary['f1']['mean']:.4f}"
    )
    print(f"[INFO] Wrote: {summary_path}")
    print(f"[INFO] Wrote: {os.path.join(args.out_dir, 'cv_aggregated_predictions.npz')}")
    print(f"[INFO] Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
