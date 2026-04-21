"""
kinship_pipeline_FUSION_ArcFace_FaceNet_HLDZP.py
────────────────────────────────────────────────────────────────────────────────
Three-modality kinship verification on KinFaceW-II
  • ArcFace   (deep features)   → L2 normalization  → diff pair feature
  • FaceNet   (deep features)   → L2 normalization  → diff pair feature
  • HLDZP     (handcrafted)     → Z-score           → diff pair feature

Each modality is processed independently through its own LocalBranch encoder.
The three 256-d embeddings are concatenated and passed to a shared fusion head.

FUSION ARCHITECTURE
────────────────────
  ArcFace  feat → L2Norm  → diff → ZScore → LocalBranch → 256-d ─┐
  FaceNet  feat → L2Norm  → diff → ZScore → LocalBranch → 256-d ──┼─ cat(768-d) → FusionHead → logit
  HLDZP    feat → ZScore  → diff →         LocalBranch → 256-d ─┘

NOTE ON PATHS
─────────────
Set the six path variables in the CONFIG section below.
ArcFace and FaceNet features are expected as .pkl files containing
either a numpy array of shape (N, feat_dim) or a list of N vectors,
one per image, in the same order as the KinFaceW .mat file indices.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — set your paths here, nothing else needs to change
# ══════════════════════════════════════════════════════════════════════════════

# ── Feature vector directories ───────────────────────────────────────────────
HLDZP_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes_classiques\Hist-LDZP\HLDZP_feature_vectors"
ARCFACE_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\ArcFace\arcface_embeddings"
FACENET_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\FaceNet\facenet_embeddings"
MAT_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

# ── Pair feature power exponent (diff normalization) ─────────────────────────
POWER_ALPHA = 0.5

# ── Training hyper-parameters ─────────────────────────────────────────────────
EPOCHS      = 300
PATIENCE    = 40
BATCH_SIZE  = 32
LR          = 5e-4
WEIGHT_DECAY= 3e-4

# ══════════════════════════════════════════════════════════════════════════════

RELATIONS = {
    "Father-Daughter": {
        "hldzp"  : os.path.join(HLDZP_DIR,   "HistLDZP_FD.pkl"),
        "arcface": os.path.join(ARCFACE_DIR,  "ArcFace_FD.pkl"),
        "facenet": os.path.join(FACENET_DIR,  "FaceNet_FD.pkl"),
        "mat"    : os.path.join(MAT_DIR,      "LBP_fd.mat"),
    },
    "Father-Son": {
        "hldzp"  : os.path.join(HLDZP_DIR,   "HistLDZP_FS.pkl"),
        "arcface": os.path.join(ARCFACE_DIR,  "ArcFace_FS.pkl"),
        "facenet": os.path.join(FACENET_DIR,  "FaceNet_FS.pkl"),
        "mat"    : os.path.join(MAT_DIR,      "LBP_fs.mat"),
    },
    "Mother-Daughter": {
        "hldzp"  : os.path.join(HLDZP_DIR,   "HistLDZP_MD.pkl"),
        "arcface": os.path.join(ARCFACE_DIR,  "ArcFace_MD.pkl"),
        "facenet": os.path.join(FACENET_DIR,  "FaceNet_MD.pkl"),
        "mat"    : os.path.join(MAT_DIR,      "LBP_md.mat"),
    },
    "Mother-Son": {
        "hldzp"  : os.path.join(HLDZP_DIR,   "HistLDZP_MS.pkl"),
        "arcface": os.path.join(ARCFACE_DIR,  "ArcFace_MS.pkl"),
        "facenet": os.path.join(FACENET_DIR,  "FaceNet_MS.pkl"),
        "mat"    : os.path.join(MAT_DIR,      "LBP_ms.mat"),
    },
}

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {DEVICE}")


# ═════════════════════════════════════════════════════════════════════════════
#  File existence check
# ═════════════════════════════════════════════════════════════════════════════

def check_files():
    print("\nChecking file paths...")
    all_ok = True
    for rel, paths in RELATIONS.items():
        for key, fpath in paths.items():
            exists = os.path.isfile(fpath)
            status = "✓" if exists else "✗ NOT FOUND"
            print(f"  [{status}]  ({key:7s})  {fpath}")
            if not exists:
                all_ok = False
    if not all_ok:
        raise FileNotFoundError(
            "\nOne or more files are missing — check the paths in the CONFIG section.\n"
            "If on Colab: from google.colab import drive; drive.mount('/content/drive')"
        )
    print("All files found.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  Feature loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_pkl(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Format 1: {filename: embedding} — new extraction scripts (ArcFace, FaceNet, etc.)
    # Keys are filenames like "fd_001_1.jpg", sorted alphabetically = correct order
    if isinstance(data, dict) and "features" not in data:
        sorted_keys = sorted(data.keys(), key=lambda k: os.path.basename(k))
        return np.array([data[k] for k in sorted_keys], dtype=np.float64)

    # Format 2: {"features": array, ...} — VGGFace2 style
    if isinstance(data, dict) and "features" in data:
        return np.array(data["features"], dtype=np.float64)

    # Format 3: plain numpy array — HistLDZP, HistLBP
    return np.array(data, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
#  Per-modality normalization
# ═════════════════════════════════════════════════════════════════════════════

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization — projects each sample onto the unit sphere."""
    return sk_normalize(X, norm="l2")


def zscore_fit_transform(X_train: np.ndarray,
                          X_test:  np.ndarray):
    """Fit StandardScaler on train, apply to both."""
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)


# ═════════════════════════════════════════════════════════════════════════════
#  Pair feature builder  (diff-only, power-normed, mean-centred)
# ═════════════════════════════════════════════════════════════════════════════

def diff_pair(feats: np.ndarray,
              idxa:  np.ndarray,
              idxb:  np.ndarray,
              alpha: float = POWER_ALPHA) -> np.ndarray:
    """
    sign(a−b) * |a−b|^alpha  −  column_mean
    Applied after per-modality normalization.
    Shape: (N, feat_dim)
    """
    a    = feats[idxa].astype(np.float64)
    b    = feats[idxb].astype(np.float64)
    diff = np.sign(a - b) * (np.abs(a - b) ** alpha)
    diff -= diff.mean(axis=0)
    return diff


# ═════════════════════════════════════════════════════════════════════════════
#  Model components
# ═════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, max(1, dim // reduction)), nn.ReLU(),
            nn.Linear(max(1, dim // reduction), dim), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.se(x)


class LocalBranch(nn.Module):
    """
    Per-modality encoder.  input_dim → 1024 → 512 → 256
    Shared architecture across all three modalities; weights are NOT shared
    (each modality gets its own LocalBranch instance).
    """
    def __init__(self, input_dim, out=256, h1=1024, h2=512,
                 d1=0.40, d2=0.30, d3=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), nn.ReLU(), SEBlock(h2), nn.Dropout(d2),
            nn.Linear(h2, out),       nn.BatchNorm1d(out), nn.ReLU(), SEBlock(out), nn.Dropout(d3),
        )
    def forward(self, x): return self.net(x)


class FusionHead(nn.Module):
    """
    768-d (3 × 256) → 512 → 256 → 128 (+residual) → 64 → 1
    """
    def __init__(self, in_dim=768):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.40),  # 0-3
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),  # 4-7
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),  # 8-11
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.15),  # 12-15
            nn.Linear(64, 1),                                                           # 16
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128),
        )

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.head):
            h = layer(h)
            if i == 10:                             # after ReLU of 256→128 block
                h = h + self.residual_proj(x)       # skip from 768-d input
        return h.squeeze(1)


class TriModalLCNN(nn.Module):
    """
    Three independent LocalBranch encoders + shared FusionHead.

    Input layout (single concatenated tensor):
      [ arcface_pair | facenet_pair | hldzp_pair ]
      dims: [arc_dim, face_dim, hldzp_dim]

    block_dims tells torch.split how to slice the input back into modalities.
    """
    def __init__(self, block_dims: list, branch_out: int = 256):
        super().__init__()
        self.block_dims = block_dims
        self.branches   = nn.ModuleList([
            LocalBranch(dim, out=branch_out) for dim in block_dims
        ])
        self.fusion = FusionHead(in_dim=branch_out * len(block_dims))

    def forward(self, x):
        parts      = torch.split(x, self.block_dims, dim=1)
        embeddings = [b(p) for b, p in zip(self.branches, parts)]
        fused      = torch.cat(embeddings, dim=1)   # (B, 768)
        return self.fusion(fused)


# ═════════════════════════════════════════════════════════════════════════════
#  Training utilities
# ═════════════════════════════════════════════════════════════════════════════

def mixup_batch(Xb, yb, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam * Xb + (1-lam) * Xb[idx], lam * yb + (1-lam) * yb[idx]


def smooth_labels(y: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    return y * (1.0 - eps) + eps * (1.0 - y)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        Xb, yb = mixup_batch(Xb, yb)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, smooth_labels(yb))
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
            preds.extend(p);  labels.extend(yb.numpy())
    return accuracy_score(labels, preds)


def train_model(X_train, y_train, X_val, y_val, block_dims):

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True,
        worker_init_fn=lambda _: np.random.seed(SEED))
    va_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False)

    model     = TriModalLCNN(block_dims=block_dims).to(DEVICE)
    pos_w     = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                             weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup: return (epoch + 1) / warmup
        return 0.5 * (1.0 + np.cos(
            np.pi * (epoch - warmup) / max(1, EPOCHS - warmup)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc, best_state, wait = 0.0, None, 0

    for epoch in range(EPOCHS):
        train_epoch(model, tr_loader, optimizer, criterion)
        scheduler.step()
        val_acc = evaluate(model, va_loader)
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop epoch {epoch+1} | "
                      f"best val={best_acc*100:.2f}%")
                break

    model.load_state_dict(best_state)
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Per-modality preprocessing pipeline
#
#  Each modality goes through:
#    1. Raw feature normalization (L2 for deep, zscore for HLDZP)
#    2. Diff pair feature construction
#    3. Final zscore on the pair vector (fit on train only)
# ═════════════════════════════════════════════════════════════════════════════

def prepare_modality(feats_raw:  np.ndarray,
                      idxa:       np.ndarray,
                      idxb:       np.ndarray,
                      norm_type:  str,            # "l2" or "zscore"
                      tr_mask:    np.ndarray,
                      te_mask:    np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X_train_pairs, X_test_pairs) for one modality, fully normalized.

    Steps
    ─────
    1. Normalize raw features (L2 or zscore fitted on ALL samples*)
       *For L2 this is sample-wise so no leakage. For zscore on raw features
        we fit on train indices only to be safe.
    2. Build diff pair vectors from normalized features.
    3. Zscore the pair vectors (fit on train pairs only).
    """
    # ── Step 1: raw feature normalization ────────────────────────────────────
    if norm_type == "l2":
        feats = l2_normalize(feats_raw.astype(np.float64))
    elif norm_type == "zscore":
        sc    = StandardScaler()
        # fit only on train image indices to avoid test leakage
        train_img_idx = np.unique(np.concatenate([idxa[tr_mask], idxb[tr_mask]]))
        sc.fit(feats_raw[train_img_idx])
        feats = sc.transform(feats_raw.astype(np.float64))
    else:
        raise ValueError(f"norm_type must be 'l2' or 'zscore', got '{norm_type}'")

    # ── Step 2: diff pair vectors ─────────────────────────────────────────────
    pairs = diff_pair(feats, idxa, idxb)

    # ── Step 3: zscore on pair vectors (fit on train pairs only) ─────────────
    X_tr_raw = pairs[tr_mask]
    X_te_raw = pairs[te_mask]
    X_train, X_test = zscore_fit_transform(X_tr_raw, X_te_raw)

    return X_train, X_test


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

check_files()

all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*70}")
    print(f"  Relation : {relation}")
    print(f"{'='*70}")

    # ── Load all three modalities ─────────────────────────────────────────────
    hldzp_feats   = load_pkl(paths["hldzp"])
    arcface_feats = load_pkl(paths["arcface"])
    facenet_feats = load_pkl(paths["facenet"])

    print(f"  HLDZP    shape : {hldzp_feats.shape}")
    print(f"  ArcFace  shape : {arcface_feats.shape}")
    print(f"  FaceNet  shape : {facenet_feats.shape}")

    # ── Load mat file (shared across modalities) ──────────────────────────────
    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    fold_scores = []

    for f in range(1, 6):
        tr_mask = fold != f
        te_mask = fold == f
        y_train = y[tr_mask]
        y_test  = y[te_mask]

        # ── Per-modality preprocessing ────────────────────────────────────────
        arc_tr, arc_te = prepare_modality(
            arcface_feats, idxa, idxb, norm_type="l2",
            tr_mask=tr_mask, te_mask=te_mask)

        face_tr, face_te = prepare_modality(
            facenet_feats, idxa, idxb, norm_type="l2",
            tr_mask=tr_mask, te_mask=te_mask)

        hldzp_tr, hldzp_te = prepare_modality(
            hldzp_feats, idxa, idxb, norm_type="zscore",
            tr_mask=tr_mask, te_mask=te_mask)

        # ── Concatenate all modalities into one tensor ────────────────────────
        X_train = np.concatenate([arc_tr,  face_tr,  hldzp_tr],  axis=1)
        X_test  = np.concatenate([arc_te,  face_te,  hldzp_te],  axis=1)

        # block_dims tells the model how to split the tensor back per modality
        block_dims = [arc_tr.shape[1], face_tr.shape[1], hldzp_tr.shape[1]]

        if f == 1:
            print(f"\n  Block dims : ArcFace={block_dims[0]}  "
                  f"FaceNet={block_dims[1]}  HLDZP={block_dims[2]}  "
                  f"Total={sum(block_dims)}")

        # ── Validation split (inner fold) ─────────────────────────────────────
        inner_fold = fold[tr_mask]
        iv_mask    = inner_fold == (f % 5 + 1)

        model = train_model(
            X_train[~iv_mask], y_train[~iv_mask],
            X_train[ iv_mask], y_train[ iv_mask],
            block_dims=block_dims,
        )

        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                          torch.tensor(y_test, dtype=torch.float32)),
            batch_size=BATCH_SIZE, shuffle=False)

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


# ─── Final summary ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FINAL SUMMARY  —  ArcFace + FaceNet + HLDZP  →  TriModal LCNN")
print(f"  Norms : ArcFace=L2 | FaceNet=L2 | HLDZP=ZScore")
print(f"  Pair  : diff-only (power α={POWER_ALPHA})")
print(f"{'='*70}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for rel, res in all_results.items():
    flag = " ✓" if res["mean_accuracy"] >= 0.88 else ""
    print(f"{rel:<22} {res['mean_accuracy']*100:>9.2f}%"
          f"  ±{res['std_accuracy']*100:>6.2f}%{flag}")

overall  = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
hldzp_bl = 74.0
deep_bl  = 88.0
print(f"\n  HLDZP alone (v1)      : {hldzp_bl:.2f}%")
print(f"  Deep baseline (LBP)   : {deep_bl:.2f}%")
print(f"  Fusion overall        : {overall*100:.2f}%")
print(f"  Gain vs HLDZP alone   : {overall*100 - hldzp_bl:+.2f}%")