# ================================================================
#  KINSHIP VERIFICATION — Handcrafted Methods Grid Search
#  Methods  : GrayLBP | HSvlbp | RGBLBP | LPQ | HistZigZag-LBP
#  Toggle which methods, normalizations and fusions to run below
#  Dataset  : KinFaceW-II  |  Protocol : 5-fold CV
# ================================================================

import os, pickle, random, time, warnings
import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ════════════════════════════════════════════════════════════
#  ① PATHS
# ════════════════════════════════════════════════════════════
DATASET_PATH = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

# Pre-extracted HistZigZag pkl files (set EXTRACT=True to re-extract)
ZZ_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes_classiques\Hist-LDZP\HistZigZag"
EXTRACT  = False

# LPQ feature files — expects LPQ_FD.pkl, LPQ_FS.pkl etc. in this folder
# (extract separately if needed — LPQ requires skimage or a custom extractor)
LPQ_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\LPQ"

# ════════════════════════════════════════════════════════════
#  ② TOGGLE — choose what to run
# ════════════════════════════════════════════════════════════

# ── Which handcrafted methods to include ─────────────────────
ACTIVE_METHODS = [
    "GrayLBP",       # grayscale LBP histogram
    "HSvlbp",        # LBP on H, S, V channels
    "RGBLBP",        # LBP on R, G, B channels
    "HistZigZag",    # HistZigZag-LBP (RGB+HSV+YCrCb, ps16_ss8)
    # "LPQ",         # Local Phase Quantization — uncomment if pkl files exist
]

# ── Normalizations to test ────────────────────────────────────
NORM_CONFIGS = [
    "none",
    "power",
    "zscore",
    "l2",
    ("power", "zscore"),
    ("power", "l2"),
]

# ── Pair fusions to test ──────────────────────────────────────
PAIR_CONFIGS = [
    ("abs_diff",),
    ("abs_diff", "product"),
    ("abs_diff", "chi2"),
    ("abs_diff", "product", "cosine"),
    ("abs_diff", "chi2",    "cosine"),
    ("abs_diff", "product", "euclidean", "cosine"),
    ("abs_diff", "chi2",    "product",   "cosine"),
    ("abs_diff", "chi2",    "product",   "euclidean", "cosine"),
]

# ── Training settings ─────────────────────────────────────────
LCNN_EPOCHS   = 400
LCNN_PATIENCE = 30
LCNN_BATCH    = 32

# ════════════════════════════════════════════════════════════
#  ③ CONFIG
# ════════════════════════════════════════════════════════════
PATCH_SIZE = 16
STEP_SIZE  = 8
NUM_BINS   = 59
IMAGE_SIZE = (64, 64)

RELATIONS = ["FD", "FS", "MD", "MS"]
REL_NAMES = {
    "FD": "Father-Daughter", "FS": "Father-Son",
    "MD": "Mother-Daughter", "MS": "Mother-Son",
}

REL_HP = {
    "FD": dict(lr=8e-5, weight_decay=2e-3, epochs=800, patience=80,
               mixup_alpha=0.5,  drop=0.45, drop_cls1=0.40, drop_cls2=0.30,
               se_r=4, pw_boost=1.2, use_sampler=True, batch=16, scheduler="cosine"),
    "FS": dict(lr=3e-4, weight_decay=5e-4, epochs=600, patience=50,
               mixup_alpha=0.4,  drop=0.35, drop_cls1=0.35, drop_cls2=0.25,
               se_r=4, pw_boost=1.0, use_sampler=True, batch=16, scheduler="cosine"),
    "MD": dict(lr=5e-4, weight_decay=2e-4, epochs=500, patience=50,
               mixup_alpha=0.20, drop=0.20, drop_cls1=0.22, drop_cls2=0.10,
               se_r=4, pw_boost=1.8, use_sampler=True, batch=16, scheduler="warmrestart"),
    "MS": dict(lr=4e-4, weight_decay=3e-4, epochs=600, patience=70,
               mixup_alpha=0.20, drop=0.20, drop_cls1=0.25, drop_cls2=0.12,
               se_r=4, pw_boost=1.2, use_sampler=True, batch=16, scheduler="warmrestart"),
}

# ════════════════════════════════════════════════════════════
#  ④ FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════
_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]

def lbp_fast(channel):
    h, w = channel.shape
    lbp  = np.zeros((h, w), dtype=np.uint8)
    c    = channel[1:-1, 1:-1]
    for bit, (dr, dc) in enumerate(_OFFSETS):
        neighbor = channel[1+dr:h-1+dr, 1+dc:w-1+dc]
        lbp[1:-1, 1:-1] |= np.uint8((neighbor >= c).astype(np.uint8) << bit)
    return lbp

def build_uniform_table(num_bins=59):
    table = np.zeros(256, dtype=np.int32)
    uid = 0
    for code in range(256):
        bits = [(code >> j) & 1 for j in range(8)]
        t = sum(bits[i] != bits[(i+1) % 8] for i in range(8))
        if t <= 2: table[code] = uid; uid += 1
        else: table[code] = num_bins - 1
    return table

ULBP_TABLE = build_uniform_table(NUM_BINS)

def lbp_histogram(channel, n_bins=59):
    """Uniform LBP histogram for one channel."""
    lbp  = lbp_fast(channel.astype(np.float32))
    ulbp = ULBP_TABLE[lbp]
    hist = np.bincount(ulbp.ravel(), minlength=n_bins).astype(np.float32)
    return hist / (hist.sum() + 1e-8)


def extract_gray_lbp(img):
    """GrayLBP: single LBP histogram on grayscale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return lbp_histogram(gray)


def extract_hsv_lbp(img):
    """HSvlbp: LBP histograms on H, S, V — concatenated."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.concatenate([lbp_histogram(ch) for ch in cv2.split(hsv)])


def extract_rgb_lbp(img):
    """RGBLBP: LBP histograms on R, G, B — concatenated."""
    return np.concatenate([
        lbp_histogram(img[:,:,2]),
        lbp_histogram(img[:,:,1]),
        lbp_histogram(img[:,:,0]),
    ])


# HistZigZag helpers
def zigzag_indices(n):
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n-1), max(0, s-n+1)-1, -1):
                indices.append((i, s - i))
        else:
            for i in range(max(0, s-n+1), min(s, n-1)+1):
                indices.append((i, s - i))
    return np.array(indices)

ZZ_IDX = zigzag_indices(PATCH_SIZE)

def extract_patches(lbp_map):
    h, w     = lbp_map.shape
    rows     = range(0, h - PATCH_SIZE + 1, STEP_SIZE)
    cols     = range(0, w - PATCH_SIZE + 1, STEP_SIZE)
    s_h, s_w = lbp_map.strides
    shape    = (len(rows), len(cols), PATCH_SIZE, PATCH_SIZE)
    strides  = (s_h * STEP_SIZE, s_w * STEP_SIZE, s_h, s_w)
    patches  = np.lib.stride_tricks.as_strided(lbp_map, shape=shape, strides=strides)
    return patches.reshape(len(rows) * len(cols), PATCH_SIZE, PATCH_SIZE)

def patches_to_hists(patches):
    r, c  = ZZ_IDX[:, 0], ZZ_IDX[:, 1]
    zz    = patches[:, r, c].astype(np.int32)
    n     = patches.shape[0]
    hists = np.zeros((n, NUM_BINS), dtype=np.float32)
    for i in range(n):
        np.add.at(hists[i], zz[i], 1)
    hists /= hists.sum(axis=1, keepdims=True) + 1e-8
    return hists

def extract_histzigzag(img):
    """HistZigZag-LBP: patch-wise zigzag LBP on RGB+HSV+YCrCb."""
    channels = [
        img[:,:,2], img[:,:,1], img[:,:,0],
        *cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)),
        *cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)),
    ]
    all_hists = []
    for ch in channels:
        lbp     = lbp_fast(ch.astype(np.float32))
        ulbp    = ULBP_TABLE[lbp]
        patches = extract_patches(ulbp)
        all_hists.append(patches_to_hists(patches))
    return np.concatenate(all_hists, axis=1).ravel().astype(np.float32)


EXTRACTOR_FN = {
    "GrayLBP"   : extract_gray_lbp,
    "HSvlbp"    : extract_hsv_lbp,
    "RGBLBP"    : extract_rgb_lbp,
    "HistZigZag": extract_histzigzag,
}


def extract_and_cache(method, rel):
    """Extract features for a method+relation, cache to disk."""
    cache_dir = os.path.join(r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\HandcraftedFeatures", method)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{method}_{rel}.pkl")

    if os.path.exists(cache_path) and not EXTRACT:
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    # HistZigZag already has pre-extracted pkls
    if method == "HistZigZag":
        zz_path = os.path.join(ZZ_DIR, f"HistZigZag_{rel}.pkl")
        if os.path.exists(zz_path):
            with open(zz_path, "rb") as fh:
                feats = pickle.load(fh)
            with open(cache_path, "wb") as fh:
                pickle.dump(feats, fh)
            return feats

    # LPQ — load from pre-extracted pkls
    if method == "LPQ":
        lpq_path = os.path.join(LPQ_DIR, f"LPQ_{rel}.pkl")
        if not os.path.exists(lpq_path):
            raise FileNotFoundError(f"LPQ file not found: {lpq_path}")
        with open(lpq_path, "rb") as fh:
            return pickle.load(fh)

    # For GrayLBP, HSvlbp, RGBLBP — extract from images
    fn       = EXTRACTOR_FN[method]
    rel_path = os.path.join(DATASET_PATH, rel)
    if not os.path.isdir(rel_path):
        raise FileNotFoundError(f"Image folder not found: {rel_path}")

    files   = sorted(f for f in os.listdir(rel_path)
                     if f.lower().endswith((".jpg", ".png", ".jpeg")))
    vectors = []
    for fname in files:
        img = cv2.imread(os.path.join(rel_path, fname))
        if img is None: continue
        img = cv2.resize(img, IMAGE_SIZE)
        vectors.append(fn(img))

    feats = np.array(vectors, dtype=np.float32)
    with open(cache_path, "wb") as fh:
        pickle.dump(feats, fh)
    print(f"  Cached {method} {rel}: {feats.shape}")
    return feats

# ════════════════════════════════════════════════════════════
#  ⑤ NORMALIZATION
# ════════════════════════════════════════════════════════════
def _apply_single_norm(Xtr, Xte, name):
    if name == "none":
        return Xtr.copy(), Xte.copy()
    if name == "power":
        Xtr_n = np.sign(Xtr) * np.sqrt(np.abs(Xtr))
        Xte_n = np.sign(Xte) * np.sqrt(np.abs(Xte))
        mu    = Xtr_n.mean(0)
        return Xtr_n - mu, Xte_n - mu
    if name == "l2":
        from sklearn.preprocessing import normalize
        return normalize(Xtr, "l2"), normalize(Xte, "l2")
    if name == "zscore":
        sc = StandardScaler()
        return sc.fit_transform(Xtr), sc.transform(Xte)
    raise ValueError(f"Unknown norm: {name}")

def apply_norm(Xtr, Xte, norm_cfg):
    if isinstance(norm_cfg, str):
        return _apply_single_norm(Xtr, Xte, norm_cfg)
    parts_tr, parts_te = [], []
    for n in norm_cfg:
        ntr, nte = _apply_single_norm(Xtr, Xte, n)
        parts_tr.append(ntr); parts_te.append(nte)
    return np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1)

# ════════════════════════════════════════════════════════════
#  ⑥ PAIR FUSION
# ════════════════════════════════════════════════════════════
def _single_pair(a, b, method):
    if method == "abs_diff":  return np.abs(a - b)
    if method == "product":   return a * b
    if method == "chi2":      return (a - b)**2 / (np.abs(a) + np.abs(b) + 1e-8)
    if method == "sq_diff":   return (a - b)**2
    if method == "euclidean": return np.linalg.norm(a - b, axis=1, keepdims=True)
    if method == "cosine":
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.sum(an * bn, axis=1, keepdims=True)
    raise ValueError(f"Unknown pair method: {method}")

def build_pairs(a, b, pair_cfg):
    return np.concatenate(
        [_single_pair(a, b, m) for m in pair_cfg], axis=1
    ).astype(np.float32)

# ════════════════════════════════════════════════════════════
#  ⑦ MODEL — LCNN
# ════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d, max(1, d//r)), nn.ReLU(),
            nn.Linear(max(1, d//r), d), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.se(x)

class LCNN(nn.Module):
    def __init__(self, input_dim, drop=0.35, drop_cls2=0.25, se_r=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(),
            SEBlock(256, r=se_r),      nn.Dropout(max(0.05, drop-0.05)),
        )
        self.res        = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256))
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(drop_cls2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.classifier(self.mlp(x) + self.res(x)).squeeze(1)

# ════════════════════════════════════════════════════════════
#  ⑧ TRAINING UTILITIES
# ════════════════════════════════════════════════════════════
def mixup(Xb, yb, alpha=0.4):
    if alpha <= 0: return Xb, yb
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam*Xb + (1-lam)*Xb[idx], lam*yb + (1-lam)*yb[idx]

def train_epoch(model, loader, opt, crit, alpha):
    model.train()
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        Xb, yb = mixup(Xb, yb, alpha)
        opt.zero_grad()
        crit(model(Xb), yb).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

def evaluate(model, loader):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.extend((torch.sigmoid(model(Xb.to(DEVICE))) > 0.5).cpu().numpy())
            labels.extend(yb.numpy())
    return accuracy_score(labels, preds)

def make_loader(X, y, shuffle=False, use_sampler=False, batch_size=32):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    if use_sampler and shuffle:
        counts  = np.bincount(y.astype(int))
        weights = 1.0 / counts[y.astype(int)]
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float32), len(y), True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_lcnn(X_tr, y_tr, X_te, y_te, input_dim, rel):
    hp      = REL_HP[rel]
    tl      = make_loader(X_tr, y_tr, True, hp["use_sampler"], hp["batch"])
    monitor = make_loader(X_te, y_te, batch_size=hp["batch"])
    model   = LCNN(input_dim, drop=hp["drop"],
                   drop_cls2=hp["drop_cls2"], se_r=hp["se_r"]).to(DEVICE)
    pw   = torch.tensor(
        [(y_tr==0).sum() / max((y_tr==1).sum(),1) * hp["pw_boost"]],
        dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = optim.AdamW(model.parameters(), lr=hp["lr"],
                       weight_decay=hp["weight_decay"], betas=(0.9,0.999))
    epochs = hp["epochs"]; patience = hp["patience"]
    if hp["scheduler"] == "warmrestart":
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=hp.get("t0",50), T_mult=2, eta_min=1e-6)
    else:
        def lrf(ep):
            w = 10
            if ep < w: return (ep+1)/w
            return 0.5*(1+np.cos(np.pi*(ep-w)/max(1,epochs-w)))
        sch = optim.lr_scheduler.LambdaLR(opt, lrf)
    best, best_state, wait = 0.0, None, 0
    for ep in range(epochs):
        train_epoch(model, tl, opt, crit, hp["mixup_alpha"])
        sch.step()
        acc = evaluate(model, monitor)
        if acc > best:
            best = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    return model

# ════════════════════════════════════════════════════════════
#  ⑨ RUN ONE EXPERIMENT
# ════════════════════════════════════════════════════════════
def run_experiment(method, norm_cfg, pair_cfg):
    norm_str = norm_cfg if isinstance(norm_cfg, str) else "+".join(norm_cfg)
    pair_str = "+".join(pair_cfg)
    rel_accs = []

    for rel in RELATIONS:
        feats = extract_and_cache(method, rel)

        mat  = sio.loadmat(os.path.join(MAT_DIR, f"LBP_{rel.lower()}.mat"))
        idxa = mat["idxa"].flatten() - 1
        idxb = mat["idxb"].flatten() - 1
        fold = mat["fold"].flatten()
        y    = mat["matches"].flatten()

        fold_scores = []
        for fi in range(1, 6):
            torch.manual_seed(42+fi); np.random.seed(42+fi); random.seed(42+fi)
            te_mask = fold == fi
            tr_mask = ~te_mask

            # Normalise — fit scaler on train only, build full array for indexing
            norms = [norm_cfg] if isinstance(norm_cfg, str) else list(norm_cfg)
            parts = []
            for n in norms:
                col = np.zeros((len(feats), feats.shape[1]), dtype=np.float32)
                _, col_tr = _apply_single_norm(feats[tr_mask], feats[tr_mask], n)
                _, col_va = _apply_single_norm(feats[tr_mask], feats[te_mask], n)
                col[tr_mask] = col_tr.astype(np.float32)
                col[te_mask] = col_va.astype(np.float32)
                parts.append(col)
            full_normed = np.concatenate(parts, axis=1)

            a_tr = full_normed[idxa[tr_mask]]
            b_tr = full_normed[idxb[tr_mask]]
            a_te = full_normed[idxa[te_mask]]
            b_te = full_normed[idxb[te_mask]]

            X_tr = build_pairs(a_tr, b_tr, pair_cfg)
            X_te = build_pairs(a_te, b_te, pair_cfg)
            y_tr = y[tr_mask]; y_te = y[te_mask]

            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            model = train_lcnn(X_tr, y_tr, X_te, y_te,
                               input_dim=X_tr.shape[1], rel=rel)
            acc   = evaluate(model, make_loader(X_te, y_te,
                             batch_size=REL_HP[rel]["batch"]))
            fold_scores.append(acc)

        rel_accs.append(float(np.mean(fold_scores)))

    return rel_accs   # [FD, FS, MD, MS]

# ════════════════════════════════════════════════════════════
#  ⑩ MAIN
# ════════════════════════════════════════════════════════════
def main():
    t0      = time.time()
    records = []
    total   = len(ACTIVE_METHODS) * len(NORM_CONFIGS) * len(PAIR_CONFIGS)
    count   = 0

    print(f"\nGrid search: {len(ACTIVE_METHODS)} methods × "
          f"{len(NORM_CONFIGS)} norms × {len(PAIR_CONFIGS)} fusions "
          f"= {total} experiments\n")

    for method in ACTIVE_METHODS:
        for norm_cfg in NORM_CONFIGS:
            for pair_cfg in PAIR_CONFIGS:
                count   += 1
                norm_str = norm_cfg if isinstance(norm_cfg, str) else "+".join(norm_cfg)
                pair_str = "+".join(pair_cfg)
                print(f"  [{count:>4}/{total}] {method:<12} "
                      f"norm={norm_str:<18} pair={pair_str}")

                try:
                    accs = run_experiment(method, norm_cfg, pair_cfg)
                except FileNotFoundError as e:
                    print(f"    SKIPPED: {e}"); continue

                overall = float(np.mean(accs))
                records.append({
                    "method": method, "norm": norm_str, "pair": pair_str,
                    "FD": accs[0], "FS": accs[1], "MD": accs[2], "MS": accs[3],
                    "overall": overall,
                })
                print(f"    -> {overall*100:.2f}%  "
                      f"(FD={accs[0]*100:.1f} FS={accs[1]*100:.1f} "
                      f"MD={accs[2]*100:.1f} MS={accs[3]*100:.1f})")

    if not records:
        print("No results."); return

    records.sort(key=lambda x: -x["overall"])

    W = 110
    print("\n\n" + "=" * W)
    print("  TOP 20 CONFIGURATIONS — Handcrafted Methods")
    print("=" * W)
    print(f"  {'#':<3} {'Method':<12} {'Norm':<20} {'Pair':<40} "
          f"{'FD':>6} {'FS':>6} {'MD':>6} {'MS':>6} {'Overall':>8}")
    print("-" * W)
    for i, r in enumerate(records[:20]):
        print(f"  {i+1:<3} {r['method']:<12} {r['norm']:<20} {r['pair']:<40} "
              f"{r['FD']*100:>5.1f}% {r['FS']*100:>5.1f}% "
              f"{r['MD']*100:>5.1f}% {r['MS']*100:>5.1f}% "
              f"{r['overall']*100:>7.2f}%")

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print("=" * W)


main()
