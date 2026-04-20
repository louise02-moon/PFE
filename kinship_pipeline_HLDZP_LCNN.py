"""
kinship_pipeline_HLDZP_LCNN_v5.py
────────────────────────────────────────────────────────────────────────────────
HistLDZP-v4  →  LCNN  (diff-only pair feature, 7-method normalization ablation)

NORM_METHOD options
───────────────────
  "zscore"    StandardScaler       — zero mean, unit variance
  "l2"        Row-wise L2          — each sample on the unit sphere
  "minmax"    MinMaxScaler         — each feature scaled to [0, 1]
  "power"     Power transform      — sign(x)*|x|^POWER_ALPHA (extra squeeze on top of diff)
  "sqrt"      Square-root          — sign(x)*√|x|  (α=0.5 special case, no scaler)
  "log"       Log1p + zscore       — log(1+|x|)*sign(x), then z-score
  "decimal"   Decimal scaling      — divide each feature by 10^k so max(|x|)≤1

Set NORM_METHOD to whichever string you want to test. Nothing else changes.

QUICK ABLATION ORDER (recommended)
────────────────────────────────────
  Run 1 :  NORM_METHOD = "zscore"     ← current baseline
  Run 2 :  NORM_METHOD = "l2"
  Run 3 :  NORM_METHOD = "power"
  Run 4 :  NORM_METHOD = "sqrt"
  Run 5 :  NORM_METHOD = "log"
  Run 6 :  NORM_METHOD = "minmax"
  Run 7 :  NORM_METHOD = "decimal"
────────────────────────────────────────────────────────────────────────────────
"""

import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                    normalize as sk_normalize)
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════════════════════
#  ▶▶  ABLATION SWITCH — change this one string to test each method  ◀◀
#
#  Options: "zscore" | "l2" | "minmax" | "power" | "sqrt" | "log" | "decimal"
# ══════════════════════════════════════════════════════════════════════════════

NORM_METHOD = "minmax"

# ── Diff-only pair feature config (fixed from v3 ablation) ───────────────────
POWER_ALPHA = 0.5    # exponent used in diff pair feature
NORM_ALPHA  = 0.4    # exponent used only when NORM_METHOD = "power"

# ══════════════════════════════════════════════════════════════════════════════

# ─── Reproducibility ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {DEVICE}")

# ─── Paths ────────────────────────────────────────────────────────────────────
HISTLDZP_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LDZP\HLDZP_feature_vectors"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "histldzp": f"{HISTLDZP_DIR}\\HistLDZP_FD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "histldzp": f"{HISTLDZP_DIR}\\HistLDZP_FS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "histldzp": f"{HISTLDZP_DIR}\\HistLDZP_MD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "histldzp": f"{HISTLDZP_DIR}\\HistLDZP_MS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_ms.mat",
    },
}

# ─── Human-readable descriptions ─────────────────────────────────────────────
NORM_DESCRIPTIONS = {
    "zscore"  : f"StandardScaler — zero mean, unit variance (z-score)",
    "l2"      :  "Row L2-norm   — each sample projected to unit sphere",
    "minmax"  :  "MinMaxScaler  — each feature scaled to [0, 1]",
    "power"   : f"Power norm    — sign(x)*|x|^{NORM_ALPHA} applied column-wise after zscore",
    "sqrt"    :  "Sqrt norm     — sign(x)*√|x|, then zscore  (α=0.5 special case)",
    "log"     :  "Log1p norm    — sign(x)*log(1+|x|), then zscore",
    "decimal" :  "Decimal scale — divide each feature by 10^k so max(|col|) ≤ 1",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Normalization  (fit on train, apply to both)
# ═════════════════════════════════════════════════════════════════════════════

def _zscore(X_tr, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def apply_normalization(X_train: np.ndarray,
                        X_test:  np.ndarray,
                        method:  str = NORM_METHOD
                        ) -> tuple[np.ndarray, np.ndarray]:
    method = method.lower().strip()

    # ── z-score ───────────────────────────────────────────────────────────────
    if method == "zscore":
        return _zscore(X_train, X_test)

    # ── L2 row-norm (after zscore to equalise feature scales first) ───────────
    elif method == "l2":
        Xtr, Xte = _zscore(X_train, X_test)
        return sk_normalize(Xtr, norm="l2"), sk_normalize(Xte, norm="l2")

    # ── Min-Max to [0, 1] ─────────────────────────────────────────────────────
    elif method == "minmax":
        sc = MinMaxScaler()
        return sc.fit_transform(X_train), sc.transform(X_test)

    # ── Power: sign(x)*|x|^alpha, fitted column-wise on train ────────────────
    elif method == "power":
        # compute per-column scale from train so test sees the same mapping
        abs_tr  = np.abs(X_train)
        col_max = abs_tr.max(axis=0) + 1e-8          # (feat_dim,) — fit on train
        def _pw(X):
            x_scaled = X / col_max                   # map roughly to [-1, 1]
            return np.sign(x_scaled) * (np.abs(x_scaled) ** NORM_ALPHA)
        Xtr = _pw(X_train)
        Xte = _pw(X_test)
        # follow with zscore so mean/variance are stable for BN layers
        return _zscore(Xtr, Xte)

    # ── Sqrt: sign(x)*√|x|, then zscore ──────────────────────────────────────
    elif method == "sqrt":
        def _sq(X):
            return np.sign(X) * np.sqrt(np.abs(X))
        return _zscore(_sq(X_train), _sq(X_test))

    # ── Log1p: sign(x)*log(1+|x|), then zscore ───────────────────────────────
    elif method == "log":
        def _lg(X):
            return np.sign(X) * np.log1p(np.abs(X))
        return _zscore(_lg(X_train), _lg(X_test))

    # ── Decimal scaling: divide each feature by 10^ceil(log10(max|col|)) ─────
    elif method == "decimal":
        col_max = np.abs(X_train).max(axis=0) + 1e-8   # fit on train only
        k       = np.ceil(np.log10(col_max))            # per-feature exponent
        scale   = 10.0 ** k                             # (feat_dim,)
        return X_train / scale, X_test / scale

    else:
        valid = list(NORM_DESCRIPTIONS.keys())
        raise ValueError(f"Unknown NORM_METHOD '{method}'. Choose from: {valid}")


# ═════════════════════════════════════════════════════════════════════════════
#  Pair feature  (diff-only, fixed)
# ═════════════════════════════════════════════════════════════════════════════

def build_pair_features(feats: np.ndarray,
                         idxa:  np.ndarray,
                         idxb:  np.ndarray,
                         alpha: float = POWER_ALPHA) -> np.ndarray:
    """
    power_norm(|a − b|, alpha) − column_mean
    Shape: (N, feat_dim)
    """
    a    = feats[idxa].astype(np.float64)
    b    = feats[idxb].astype(np.float64)
    diff = np.sign(a - b) * (np.abs(a - b) ** alpha)   # power-normed diff
    diff -= diff.mean(axis=0)
    return diff


# ═════════════════════════════════════════════════════════════════════════════
#  Model
# ═════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, max(1, dim // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, dim // reduction), dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class LocalBranch(nn.Module):
    def __init__(self, input_dim, out=256,
                 h1=1024, h2=512,
                 d1=0.40, d2=0.30, d3=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(d1),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            SEBlock(h2),
            nn.Dropout(d2),

            nn.Linear(h2, out),
            nn.BatchNorm1d(out),
            nn.ReLU(),
            SEBlock(out),
            nn.Dropout(d3),
        )

    def forward(self, x):
        return self.net(x)


class LCNN(nn.Module):
    def __init__(self, block_dims, local_out=256):
        super().__init__()
        self.block_dims = block_dims
        self.local_out  = local_out

        self.local_branches = nn.ModuleList([
            LocalBranch(dim, out=local_out) for dim in block_dims
        ])

        fusion_dim = local_out * len(block_dims)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 1),
        )

        self.residual_proj = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        splits     = torch.split(x, self.block_dims, dim=1)
        local_outs = [b(s) for b, s in zip(self.local_branches, splits)]
        fused      = torch.cat(local_outs, dim=1)

        # manual forward to inject residual after layer index 6
        h = fused
        for i, layer in enumerate(self.fusion):
            h = layer(h)
            if i == 6:                          # after ReLU of second block
                h = h + self.residual_proj(fused)

        return h.squeeze(1)


# ═════════════════════════════════════════════════════════════════════════════
#  Training utilities
# ═════════════════════════════════════════════════════════════════════════════

def mixup_batch(Xb, yb, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam * Xb + (1 - lam) * Xb[idx], lam * yb + (1 - lam) * yb[idx]


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        Xb, yb = mixup_batch(Xb, yb)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += ((torch.sigmoid(logits) >= 0.5).long()
                       == yb.long()).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            p = (torch.sigmoid(model(Xb.to(DEVICE))) >= 0.5).long().cpu().numpy()
            preds.extend(p)
            labels.extend(yb.numpy())
    return accuracy_score(labels, preds)


def train_lcnn(X_train, y_train, X_val, y_val, block_dims,
               epochs=300, batch_size=32, lr=5e-4, patience=30):

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
        worker_init_fn=lambda _: np.random.seed(42))

    va_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False)

    model     = LCNN(block_dims=block_dims).to(DEVICE)
    pos_w     = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=2e-4, betas=(0.9, 0.999))

    def lr_lambda(epoch):
        warmup   = 10
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler  = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc, best_state, wait = 0.0, None, 0

    for epoch in range(epochs):
        train_epoch(model, tr_loader, optimizer, criterion)
        scheduler.step()
        val_acc = evaluate(model, va_loader)
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop epoch {epoch+1} | "
                      f"best val={best_acc*100:.2f}%")
                break

    model.load_state_dict(best_state)
    return model


# ─── Feature loader ───────────────────────────────────────────────────────────
def load_histldzp(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data, dtype=np.float64)


# ─── Startup config printout ──────────────────────────────────────────────────
def print_config(feat_dim: int) -> None:
    print(f"\n{'─'*65}")
    print(f"  Pipeline config  (v5 — normalization ablation)")
    print(f"{'─'*65}")
    print(f"  Pair feature   : diff-only  (power α={POWER_ALPHA})")
    print(f"  Pair vec dim   : {feat_dim:,}")
    print(f"  Norm method    : {NORM_METHOD}")
    print(f"  Description    : {NORM_DESCRIPTIONS.get(NORM_METHOD, 'unknown')}")
    print(f"{'─'*65}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*70}")
    print(f"  Relation : {relation}")
    print(f"{'='*70}")

    feats = load_histldzp(paths["histldzp"])
    print(f"  HistLDZP shape : {feats.shape}")

    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    pairs      = build_pair_features(feats, idxa, idxb)
    block_dims = [pairs.shape[1]]
    print_config(pairs.shape[1])
    print(f"  Pairs shape : {pairs.shape}  |  block_dims : {block_dims}")

    fold_scores = []

    for f in range(1, 6):
        tr_mask = fold != f
        te_mask = fold == f

        X_train_raw = pairs[tr_mask]
        X_test_raw  = pairs[te_mask]
        y_train     = y[tr_mask]
        y_test      = y[te_mask]

        X_train, X_test = apply_normalization(X_train_raw, X_test_raw)

        inner_fold = fold[tr_mask]
        iv_mask    = inner_fold == (f % 5 + 1)

        model = train_lcnn(
            X_train[~iv_mask], y_train[~iv_mask],
            X_train[ iv_mask], y_train[ iv_mask],
            block_dims = block_dims,
            epochs     = 300,
            batch_size = 32,
            lr         = 5e-4,
            patience   = 30,
        )

        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                          torch.tensor(y_test, dtype=torch.float32)),
            batch_size=32, shuffle=False)

        acc = evaluate(model, test_loader)
        fold_scores.append(acc)
        print(f"  Fold {f}: {acc*100:.2f}%")

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))
    all_results[relation] = {
        "fold_scores"  : fold_scores,
        "mean_accuracy": mean_acc,
        "std_accuracy" : std_acc,
    }
    print(f"  ── Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")


# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FINAL SUMMARY  —  HistLDZP → LCNN")
print(f"  Pair: diff-only (α={POWER_ALPHA})   Norm: {NORM_METHOD}")
print(f"{'='*70}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for rel, res in all_results.items():
    print(f"{rel:<22} {res['mean_accuracy']*100:>9.2f}%"
          f"  ±{res['std_accuracy']*100:>6.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
prev    = 73.90
print(f"\n  Norm method         : {NORM_METHOD}")
print(f"  Previous (v1)       : {prev:.2f}%")
print(f"  Overall Accuracy    : {overall*100:.2f}%")
