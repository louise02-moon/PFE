"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   KINSHIP VERIFICATION — LCNN + HistZigZag LBP                             ║
║   ArcFace · FaceNet · ResNet50 · VGGFace · HistZigZagLBP                  ║
║   5-Fold Cross-Validation · MixUp · SE Attention · Residual · Warmup LR   ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ WHAT CHANGED vs. YOUR HistLBP VERSION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] LOADER — NumpyCompatUnpickler
    Handles numpy version mismatches (the _reconstruct error) automatically.
    Tries 3 strategies: compat unpickler → latin1 → bytes encoding.

[2] HISTZIGZAG PAIR FEATURES (replaces build_hlbp_pair_features)
    Your old HistLBP used: sqrt(|a-b|) - mean  (1 component, dim=D)
    New HistZigZag uses 3 complementary similarity scores (dim=3):
      • Cosine similarity        → direction agreement between histograms
      • Negative L2 distance     → absolute closeness
      • Negative Chi-squared     → standard histogram bin comparison
    Much more compact and discriminative for ZigZag-ordered LBP histograms.

[3] DEEP PAIR FEATURES — 5 components instead of 3
    Old: [|a-b|, a*b, ‖a-b‖]            dim = 2D+1
    New: [a-b, |a-b|, a*b, (a-b)², cos] dim = 5D
    Signed diff captures direction; squared diff amplifies large gaps;
    tiled cosine gives a global similarity signal per dimension.

[4] LABEL SMOOTHING properly applied (ε=0.05)
    Targets are smoothed: 0→0.025, 1→0.975 before BCEWithLogitsLoss.

[5] GRADIENT CLIPPING reduced: 2.0 → 1.0 (more stable)

[6] DROPOUT reduced: 0.35/0.4 → 0.15/0.20 (less aggressive regularisation,
    better for already-compact fused features)

[7] Everything else (5-fold CV, inner val, warmup+cosine scheduler,
    early stopping, MixUp, SE blocks, residual connection) preserved exactly.
"""

import pickle, random, sys
import numpy as np
import numpy.core
import numpy.core.multiarray
import scipy.io as sio
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS  — update if needed
# ══════════════════════════════════════════════════════════════════════════════
ARCFACE_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\ArcFace\arcface_embeddings"
FACENET_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\FaceNet\facenet_embeddings"
RESNET50_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\ResNet50\resnet50_embeddings"
VGGFACE_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\VGGFace\vggface_embeddings"
HISTZIGZAG_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes_classiques\Hist-LDZP\HLDZP_feature_vectors"
MAT_DIR       = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "arcface":    f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "facenet":    f"{FACENET_DIR}\\FaceNet_FD.pkl",
        "resnet50":   f"{RESNET50_DIR}\\ResNet50_FD.pkl",
        "vggface":    f"{VGGFACE_DIR}\\VGGFace_FD.pkl",
        "histzigzag": f"{HISTZIGZAG_DIR}\\HistLDZP_FD.pkl",
        "mat":        f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "arcface":    f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "facenet":    f"{FACENET_DIR}\\FaceNet_FS.pkl",
        "resnet50":   f"{RESNET50_DIR}\\ResNet50_FS.pkl",
        "vggface":    f"{VGGFACE_DIR}\\VGGFace_FS.pkl",
        "histzigzag": f"{HISTZIGZAG_DIR}\\HistLDZP_FS.pkl",
        "mat":        f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "arcface":    f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "facenet":    f"{FACENET_DIR}\\FaceNet_MD.pkl",
        "resnet50":   f"{RESNET50_DIR}\\ResNet50_MD.pkl",
        "vggface":    f"{VGGFACE_DIR}\\VGGFace_MD.pkl",
        "histzigzag": f"{HISTZIGZAG_DIR}\\HistLDZP_MD.pkl",
        "mat":        f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "arcface":    f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "facenet":    f"{FACENET_DIR}\\FaceNet_MS.pkl",
        "resnet50":   f"{RESNET50_DIR}\\ResNet50_MS.pkl",
        "vggface":    f"{VGGFACE_DIR}\\VGGFace_MS.pkl",
        "histzigzag": f"{HISTZIGZAG_DIR}\\HistLDZP_MS.pkl",
        "mat":        f"{MAT_DIR}\\LBP_ms.mat",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# ROBUST LOADER  (fixes numpy version mismatch / _reconstruct error)
# ══════════════════════════════════════════════════════════════════════════════
class NumpyCompatUnpickler(pickle.Unpickler):
    """Redirects broken numpy.core / numpy._core references to current numpy."""
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray',
                      'numpy._core', 'numpy.core'):
            for ns in (np, np.core, np.core.multiarray):
                if hasattr(ns, name):
                    return getattr(ns, name)
        if module.startswith('numpy._core'):
            try:
                return super().find_class(
                    module.replace('numpy._core', 'numpy.core'), name)
            except Exception:
                pass
        return super().find_class(module, name)


def _raw_load(path: str):
    for strategy, fn in [
        ("compat",  lambda: NumpyCompatUnpickler(open(path, "rb")).load()),
        ("latin1",  lambda: pickle.load(open(path, "rb"), encoding="latin1")),
        ("bytes",   lambda: pickle.load(open(path, "rb"), encoding="bytes")),
    ]:
        try:
            return fn()
        except Exception:
            pass
    raise RuntimeError(f"Cannot load pickle: {path}")


def _to_float64(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float64)
    if isinstance(data, dict):
        if "features" in data:
            return np.array(data["features"], dtype=np.float64)
        vals = list(data.values())
        # dict of {filename: embedding}  — sort by key for reproducibility
        if isinstance(list(data.keys())[0], str):
            vals = [data[k] for k in sorted(data.keys())]
        return np.vstack(vals).astype(np.float64)
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            for key in ("embedding", "feat", "feature"):
                if key in data[0]:
                    return np.vstack([d[key] for d in data]).astype(np.float64)
        return np.vstack(data).astype(np.float64)
    return np.array(data, dtype=np.float64)


def load_deep_embedding(path: str) -> np.ndarray:
    """Load deep embedding, L2-normalise to unit sphere."""
    data  = _raw_load(path)
    feats = _to_float64(data)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / (norms + 1e-8)


def load_histzigzag(path: str) -> np.ndarray:
    """Load HistZigZagLBP feature matrix (no normalisation — kept as histogram)."""
    data = _raw_load(path)
    return _to_float64(data)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def power_normalize(X: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """sign(x) * |x|^alpha — compresses dynamic range."""
    return np.sign(X) * (np.abs(X) ** alpha)


def build_deep_pair_features(feats: np.ndarray, idxa, idxb,
                              alpha: float = 0.5) -> np.ndarray:
    """
    5-component pairwise descriptor for deep embeddings.
    Embeddings must already be L2-normalised (done in load_deep_embedding).

      Component       Formula          Dim      Why
      ─────────────── ──────────────── ──────── ──────────────────────────────
      signed diff     a - b            D        directional shift
      absolute diff   |a - b|          D        magnitude (most discriminative)
      hadamard prod   a ⊙ b            D        co-activation pattern
      squared diff    (a - b)²         D        amplifies large gaps
      cosine (tiled)  cos(a,b) × 1_D  D        global similarity per dim

    All components power-normalised (α=0.5) before concat.
    Output dim: 5 × embed_dim
    """
    a, b = feats[idxa], feats[idxb]
    eps  = 1e-8

    signed_diff  = power_normalize(a - b,          alpha)
    abs_diff     = power_normalize(np.abs(a - b),  alpha)
    hadamard     = power_normalize(a * b,           alpha)
    sq_diff      = power_normalize((a - b) ** 2,   alpha)

    cosine_val   = (a * b).sum(axis=1, keepdims=True)          # (N,1)
    cosine_tiled = power_normalize(
        np.tile(cosine_val, (1, feats.shape[1])), alpha)       # (N,D)

    return np.concatenate(
        [signed_diff, abs_diff, hadamard, sq_diff, cosine_tiled],
        axis=1
    ).astype(np.float32)


def build_histzigzag_pair_features(feats: np.ndarray,
                                    idxa, idxb) -> np.ndarray:
    """
    3-score compact descriptor for HistZigZagLBP histograms.

    Why NOT raw concatenation?
      ZigZag LBP histograms can be 1000+ dimensional and very sparse.
      Raw [a, b] or [|a-b|] would create a huge, noisy feature vector.
      Instead we compute 3 complementary similarity metrics that are
      compact (dim=3), interpretable, and histogram-aware:

      Score           Formula                      Meaning
      ─────────────── ──────────────────────────── ─────────────────────────
      cosine sim      (a/‖a‖)·(b/‖b‖)             direction agreement
      negative L2     -‖a - b‖                    absolute closeness
      negative Chi²   -Σ(ap-bp)²/(ap+bp+ε)        bin-level comparison
                      (ap = power_norm(|a|))

    Output dim: 3
    """
    eps = 1e-8
    a, b = feats[idxa], feats[idxb]

    na = np.linalg.norm(a, axis=1, keepdims=True) + eps
    nb = np.linalg.norm(b, axis=1, keepdims=True) + eps
    cosine = ((a / na) * (b / nb)).sum(axis=1, keepdims=True)

    l2_neg = -np.linalg.norm(a - b, axis=1, keepdims=True)

    ap, bp   = power_normalize(np.abs(a)), power_normalize(np.abs(b))
    chi2_neg = -(((ap - bp) ** 2) / (ap + bp + eps)).sum(axis=1, keepdims=True)

    return np.concatenate([cosine, l2_neg, chi2_neg], axis=1).astype(np.float32)


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    """StandardScaler fit on train only — no leakage."""
    sc = StandardScaler()
    return (sc.fit_transform(X_train).astype(np.float32),
            sc.transform(X_test).astype(np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns per-feature importance weights."""
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, max(1, dim // reduction)), nn.ReLU(),
            nn.Linear(max(1, dim // reduction), dim), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x)


class LocalBranch(nn.Module):
    """
    Per-modality branch: keeps each modality's representation separate
    before fusion, preventing dominant modalities from overwhelming others.
    Dropout reduced vs. original (0.35→0.15, 0.2→0.10) since features
    are already compact after pair construction.
    """
    def __init__(self, input_dim: int, hidden: int = 512,
                 out: int = 256, drop1: float = 0.15, drop2: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(drop1),
            nn.Linear(hidden, out),
            nn.BatchNorm1d(out),   nn.ReLU(),
            SEBlock(out),
            nn.Dropout(drop2),
        )
    def forward(self, x):
        return self.net(x)


class LCNN(nn.Module):
    """
    LCNN with:
    - One LocalBranch (SE attention) per modality
    - Residual connection at the 512-dim fusion layer
    - Deep head: 1280→768→512→256→128→1
    - Label smoothing applied externally (see SmoothedBCE)
    """
    def __init__(self, block_dims: list, local_out: int = 256):
        super().__init__()
        self.block_dims = block_dims
        self.local_out  = local_out

        self.local_branches = nn.ModuleList([
            LocalBranch(dim, hidden=512, out=local_out, drop1=0.15, drop2=0.10)
            for dim in block_dims
        ])

        fusion_dim = local_out * len(block_dims)   # 256×5 = 1280

        self.residual_proj = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 768), nn.BatchNorm1d(768),   # 0,1
            nn.ReLU(), nn.Dropout(0.25),                       # 2,3

            nn.Linear(768, 512), nn.BatchNorm1d(512),          # 4,5
            nn.ReLU(), nn.Dropout(0.20),                       # 6,7

            nn.Linear(512, 256), nn.BatchNorm1d(256),          # 8,9
            nn.ReLU(), nn.Dropout(0.15),                       # 10,11

            nn.Linear(256, 128), nn.BatchNorm1d(128),          # 12,13
            nn.ReLU(), nn.Dropout(0.10),                       # 14,15

            nn.Linear(128, 1),                                  # 16
        )

    def forward(self, x):
        splits     = torch.split(x, self.block_dims, dim=1)
        local_outs = [b(s) for b, s in zip(self.local_branches, splits)]
        fused      = torch.cat(local_outs, dim=1)   # (B, 1280)

        # Fusion with residual at the 512-dim layer
        h = self.fusion[0](fused)   # Linear → 768
        h = self.fusion[1](h)       # BN
        h = self.fusion[2](h)       # ReLU
        h = self.fusion[3](h)       # Dropout

        h = self.fusion[4](h)       # Linear → 512
        h = self.fusion[5](h)       # BN
        h = self.fusion[6](h)       # ReLU
        h = h + self.residual_proj(fused)   # ← residual
        h = self.fusion[7](h)       # Dropout

        h = self.fusion[8](h)       # Linear → 256
        h = self.fusion[9](h)       # BN
        h = self.fusion[10](h)      # ReLU
        h = self.fusion[11](h)      # Dropout

        h = self.fusion[12](h)      # Linear → 128
        h = self.fusion[13](h)      # BN
        h = self.fusion[14](h)      # ReLU
        h = self.fusion[15](h)      # Dropout

        return self.fusion[16](h).squeeze(1)   # (B,)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
class SmoothedBCELoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing ε.
    Replaces hard 0/1 targets with ε/2 and 1-ε/2.
    Prevents overconfident predictions and improves generalisation.
    """
    def __init__(self, pos_weight=None, eps: float = 0.05):
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def forward(self, logits, targets):
        smooth = targets * (1 - self.eps) + (1 - targets) * self.eps
        return self.bce(logits, smooth)


def mixup_batch(Xb, yb, alpha: float = 0.3):
    """MixUp augmentation: convex interpolation of two training examples."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam * Xb + (1 - lam) * Xb[idx], lam * yb + (1 - lam) * yb[idx]


def train_epoch(model, loader, optimizer, criterion, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        if use_mixup:
            Xb, yb = mixup_batch(Xb, yb, alpha=0.3)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 2.0→1.0
        optimizer.step()
        total_loss += loss.item() * len(yb)
        preds    = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == yb.long()).sum().item()
        total   += len(yb)
    return total_loss / total, correct / total


def evaluate_loader(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            probs = torch.sigmoid(model(Xb)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    return acc, auc, np.array(all_preds), np.array(all_labels)


def train_lcnn(X_train, y_train, X_val, y_val, block_dims,
               epochs=250, batch_size=32, lr=5e-4, patience=25):

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_va = torch.tensor(X_val,   dtype=torch.float32)
    y_va = torch.tensor(y_val,   dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size,
        shuffle=True, worker_init_fn=lambda _: np.random.seed(42))
    val_loader   = DataLoader(
        TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)

    model = LCNN(block_dims=block_dims).to(DEVICE)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32).to(DEVICE)

    criterion = SmoothedBCELoss(pos_weight=pos_weight, eps=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=2e-4, betas=(0.9, 0.999))

    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc, best_state, wait = 0.0, None, 0

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, use_mixup=True)
        scheduler.step()
        val_acc, _, _, _ = evaluate_loader(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch+1} | "
                      f"best val={best_val_acc*100:.2f}%")
                break

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — 5-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*70}")
    print(f"  Relation : {relation}")
    print(f"{'='*70}")

    arcface_feats  = load_deep_embedding(paths["arcface"])
    facenet_feats  = load_deep_embedding(paths["facenet"])
    resnet50_feats = load_deep_embedding(paths["resnet50"])
    vggface_feats  = load_deep_embedding(paths["vggface"])
    histzz_feats   = load_histzigzag(paths["histzigzag"])

    print(f"  ArcFace     : {arcface_feats.shape}")
    print(f"  FaceNet     : {facenet_feats.shape}")
    print(f"  ResNet50    : {resnet50_feats.shape}")
    print(f"  VGGFace     : {vggface_feats.shape}")
    print(f"  HistZigZag  : {histzz_feats.shape}")

    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    arc_pairs = build_deep_pair_features(arcface_feats,  idxa, idxb)
    fn_pairs  = build_deep_pair_features(facenet_feats,  idxa, idxb)
    rn_pairs  = build_deep_pair_features(resnet50_feats, idxa, idxb)
    vgg_pairs = build_deep_pair_features(vggface_feats,  idxa, idxb)
    hzz_pairs = build_histzigzag_pair_features(histzz_feats, idxa, idxb)

    block_dims = [
        arc_pairs.shape[1],
        fn_pairs.shape[1],
        rn_pairs.shape[1],
        vgg_pairs.shape[1],
        hzz_pairs.shape[1],   # always 3
    ]

    X = np.concatenate([arc_pairs, fn_pairs, rn_pairs, vgg_pairs, hzz_pairs], axis=1)
    print(f"  Combined    : {X.shape}  |  Blocks: {block_dims}")

    fold_scores, fold_aucs = [], []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y[train_mask]
        y_test      = y[test_mask]

        X_train_sc, X_test_sc = normalize(X_train_raw, X_test_raw)

        # Inner validation: use fold (f%5)+1 as inner val set
        inner_val_mask  = fold[train_mask] == (f % 5 + 1)
        X_inner_tr = X_train_sc[~inner_val_mask]
        X_inner_va = X_train_sc[ inner_val_mask]
        y_inner_tr = y_train[~inner_val_mask]
        y_inner_va = y_train[ inner_val_mask]

        model = train_lcnn(
            X_inner_tr, y_inner_tr,
            X_inner_va, y_inner_va,
            block_dims=block_dims,
            epochs=250, batch_size=32, lr=5e-4, patience=25,
        )

        X_test_t  = torch.tensor(X_test_sc,  dtype=torch.float32)
        y_test_t  = torch.tensor(y_test,     dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t),
                                 batch_size=32, shuffle=False)
        acc, auc, preds, labels = evaluate_loader(model, test_loader)
        fold_scores.append(acc)
        fold_aucs.append(auc)
        print(f"  Fold {f}: Acc={acc*100:.2f}%  AUC={auc:.4f}")

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))
    mean_auc = float(np.mean(fold_aucs))
    all_results[relation] = {
        "fold_scores": fold_scores, "fold_aucs": fold_aucs,
        "mean_accuracy": mean_acc,  "std_accuracy": std_acc,
        "mean_auc": mean_auc,
    }
    print(f"  ── Mean Acc : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%  |  "
          f"Mean AUC: {mean_auc:.4f}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  FINAL SUMMARY  —  LCNN + HistZigZagLBP")
print(f"{'='*70}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8} {'AUC':>8}")
print("-" * 52)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%"
          f" {res['mean_auc']:>8.4f}")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall Mean Accuracy : {overall*100:.2f}%")
print(f"  Baseline (HistLBP)    : 88.65%")
print(f"  Gain                  : {(overall - 0.8865)*100:+.2f}%")
print(f"{'='*70}")