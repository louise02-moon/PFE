"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HistZigZag LBP — Grid Search Pipeline (VS Code / Windows)               ║
║  Multiple configs × normalizations × pair fusions → LCNN                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2, os, pickle, random
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize as sk_normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

random.seed(42); np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# ①  USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
OUTPUT_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes_classiques\Hist-LDZP\HistZigZag"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

EXTRACT = False    # True = extract, False = skip to grid search

IMAGE_SIZE = (64, 64)
NUM_BINS   = 59

EXTRACT_CONFIGS = [
    #{"patch_size": 32, "step_size": 4,  "tag": "ps32_ss4"},
    {"patch_size": 16, "step_size": 8,  "tag": "ps16_ss8"},
    #{"patch_size": 32, "step_size": 2,  "tag": "ps32_ss2"},
]

NORM_CONFIGS = [
    "power",
    "l2",
    "zscore",
    #("power", "l2"),
    #("power", "zscore"),
]

PAIR_CONFIGS = [
    "abs_diff",
    ("abs_diff", "product"),
    ("abs_diff", "product", "cosine"),
    ("abs_diff", "product", "euclidean", "cosine"),
]

LCNN_EPOCHS   = 350
LCNN_PATIENCE = 25
LCNN_LR       = 5e-4
LCNN_BATCH    = 32

RELATIONS = ["FD", "FS", "MD", "MS"]
REL_NAMES = {"FD": "Father-Daughter", "FS": "Father-Son",
             "MD": "Mother-Daughter", "MS": "Mother-Son"}

# ══════════════════════════════════════════════════════════════════════════════
# ②  ZIGZAG LBP
# ══════════════════════════════════════════════════════════════════════════════

def build_uniform_table():
    table = np.zeros(256, dtype=np.int32)
    uid = 0
    for code in range(256):
        bits = [(code >> j) & 1 for j in range(8)]
        t = sum(bits[i] != bits[(i+1) % 8] for i in range(8))
        if t <= 2: table[code] = uid; uid += 1
        else: table[code] = NUM_BINS - 1
    return table

ULBP_TABLE = build_uniform_table()

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

_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]

def lbp_fast(channel):
    h, w = channel.shape
    lbp  = np.zeros((h, w), dtype=np.uint8)
    c    = channel[1:-1, 1:-1]
    for bit, (dr, dc) in enumerate(_OFFSETS):
        neighbor = channel[1+dr:h-1+dr, 1+dc:w-1+dc]
        lbp[1:-1, 1:-1] |= np.uint8((neighbor >= c).astype(np.uint8) << bit)
    return lbp

def extract_patches(lbp_map, patch_size, step_size):
    h, w     = lbp_map.shape
    rows     = range(0, h - patch_size + 1, step_size)
    cols     = range(0, w - patch_size + 1, step_size)
    s_h, s_w = lbp_map.strides
    shape    = (len(rows), len(cols), patch_size, patch_size)
    strides  = (s_h * step_size, s_w * step_size, s_h, s_w)
    patches  = np.lib.stride_tricks.as_strided(lbp_map, shape=shape, strides=strides)
    return patches.reshape(len(rows) * len(cols), patch_size, patch_size)

def patches_to_hists(patches, zz_idx):
    r, c  = zz_idx[:, 0], zz_idx[:, 1]
    zz    = patches[:, r, c].astype(np.int32)
    N     = patches.shape[0]
    hists = np.zeros((N, NUM_BINS), dtype=np.float32)
    for i in range(N):
        np.add.at(hists[i], zz[i], 1)
    hists /= (hists.sum(axis=1, keepdims=True) + 1e-8)
    return hists

def extract_histzigzag(image_path, patch_size, step_size):
    img = cv2.imread(image_path)
    if img is None: return None

    img = cv2.resize(img, IMAGE_SIZE)
    channels = [
        img[:,:,2], img[:,:,1], img[:,:,0],
        *cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)),
        *cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)),
    ]

    zz_idx = zigzag_indices(patch_size)
    all_hists = []
    for ch in channels:
        lbp     = lbp_fast(ch.astype(np.float32))
        ulbp    = ULBP_TABLE[lbp]
        patches = extract_patches(ulbp, patch_size, step_size)
        all_hists.append(patches_to_hists(patches, zz_idx))

    return np.concatenate(all_hists, axis=1).ravel().astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# ③  EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cfg in EXTRACT_CONFIGS:
        tag     = cfg["tag"]
        cfg_dir = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"\nExtracting: {tag}  (patch={cfg['patch_size']} step={cfg['step_size']})")
        for rel in RELATIONS:
            rel_path = os.path.join(DATASET_PATH, rel)
            files = sorted([f for f in os.listdir(rel_path)
                            if f.lower().endswith((".jpg",".png",".jpeg"))])
            vectors = []
            for k, fname in enumerate(files):
                vec = extract_histzigzag(os.path.join(rel_path, fname),
                                         cfg["patch_size"], cfg["step_size"])
                if vec is not None: vectors.append(vec)
                if (k+1) % 100 == 0: print(f"    {rel} {k+1}/{len(files)}")
            vectors = np.array(vectors, dtype=np.float32)
            with open(os.path.join(cfg_dir, f"HistZigZag_{rel}.pkl"), "wb") as f:
                pickle.dump(vectors, f)
            print(f"    {rel}: {vectors.shape}")
    print("\nExtraction complete.")

# ══════════════════════════════════════════════════════════════════════════════
# ④  NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _single_norm(Xtr, Xte, name):
    if name == "power":
        Xtr_n = np.sqrt(np.abs(Xtr)); mu = Xtr_n.mean(0)
        return Xtr_n - mu, np.sqrt(np.abs(Xte)) - mu
    if name == "l1":
        return (Xtr / (np.abs(Xtr).sum(1, keepdims=True) + 1e-8),
                Xte / (np.abs(Xte).sum(1, keepdims=True) + 1e-8))
    if name == "l2":
        return sk_normalize(Xtr, "l2"), sk_normalize(Xte, "l2")
    if name == "zscore":
        sc = StandardScaler(); return sc.fit_transform(Xtr), sc.transform(Xte)
    if name == "minmax":
        sc = MinMaxScaler(); return sc.fit_transform(Xtr), sc.transform(Xte)
    if name == "none":
        return Xtr, Xte
    raise ValueError(f"Unknown norm: {name}")

def apply_norm(Xtr, Xte, norm_cfg):
    if isinstance(norm_cfg, str):
        return _single_norm(Xtr, Xte, norm_cfg)
    parts_tr, parts_te = [], []
    for n in norm_cfg:
        ntr, nte = _single_norm(Xtr, Xte, n)
        parts_tr.append(ntr); parts_te.append(nte)
    return np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1)

# ══════════════════════════════════════════════════════════════════════════════
# ⑤  PAIR FUSION
# ══════════════════════════════════════════════════════════════════════════════

def _single_pair(feats, idxa, idxb, method):
    a, b = feats[idxa], feats[idxb]
    if method == "abs_diff":  return np.abs(a - b)
    if method == "product":   return a * b
    if method == "sq_diff":   return (a - b) ** 2
    if method == "sum":       return (a + b) / 2.0
    if method == "euclidean":
        return np.linalg.norm(a - b, axis=1, keepdims=True)
    if method == "cosine":
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.sum(an * bn, axis=1, keepdims=True)
    raise ValueError(f"Unknown pair: {method}")

def build_pairs(feats, idxa, idxb, pair_cfg):
    if isinstance(pair_cfg, str):
        return _single_pair(feats, idxa, idxb, pair_cfg)
    return np.concatenate([_single_pair(feats, idxa, idxb, m)
                           for m in pair_cfg], axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# ⑥  LCNN
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.se = nn.Sequential(nn.Linear(d, max(1,d//r)), nn.ReLU(),
                                nn.Linear(max(1,d//r), d), nn.Sigmoid())
    def forward(self, x): return x * self.se(x)

class LCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(),
            SEBlock(256),              nn.Dropout(0.25),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
        )
        self.skip = nn.Sequential(nn.Linear(input_dim, 128), nn.BatchNorm1d(128))
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.branch(x) + self.skip(x)).squeeze(1)

def mixup(Xb, yb, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam*Xb+(1-lam)*Xb[idx], lam*yb+(1-lam)*yb[idx]

def train_ep(m, ld, opt, crit):
    m.train()
    for Xb, yb in ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE); Xb, yb = mixup(Xb, yb)
        opt.zero_grad(); crit(m(Xb), yb).backward()
        nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()

def evaluate(m, ld):
    m.eval(); p, l = [], []
    with torch.no_grad():
        for Xb, yb in ld:
            p.extend((torch.sigmoid(m(Xb.to(DEVICE))) > 0.5).cpu().numpy())
            l.extend(yb.numpy())
    return accuracy_score(l, p)

def make_loader(X, y, shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.float32)),
                      batch_size=LCNN_BATCH, shuffle=shuffle)

def split_fold_half(fold_mask, seed=42):
    indices = np.where(fold_mask)[0]
    rng = np.random.RandomState(seed); rng.shuffle(indices)
    mid = len(indices) // 2
    mask_te = np.zeros(len(fold_mask), dtype=bool)
    mask_va = np.zeros(len(fold_mask), dtype=bool)
    mask_te[indices[:mid]] = True; mask_va[indices[mid:]] = True
    return mask_te, mask_va

def train_lcnn(X_tr, y_tr, X_va, y_va, input_dim):
    tl = make_loader(X_tr, y_tr, shuffle=True)
    vl = make_loader(X_va, y_va)
    m  = LCNN(input_dim).to(DEVICE)
    pw = torch.tensor([(y_tr==0).sum()/max((y_tr==1).sum(),1)],
                      dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = optim.AdamW(m.parameters(), lr=LCNN_LR, weight_decay=2e-4, betas=(0.9, 0.999))

    def lrf(ep):
        warmup = 10
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+np.cos(np.pi*(ep-warmup)/max(1, LCNN_EPOCHS-warmup)))
    sch = optim.lr_scheduler.LambdaLR(opt, lrf)

    best, state, wait = 0.0, None, 0
    for ep in range(LCNN_EPOCHS):
        train_ep(m, tl, opt, crit); sch.step()
        acc = evaluate(m, vl)
        if acc > best:
            best = acc; state = {k:v.cpu().clone() for k,v in m.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= LCNN_PATIENCE: break
    m.load_state_dict(state); return m

# ══════════════════════════════════════════════════════════════════════════════
# ⑦  RUN ONE EXPERIMENT — 5-fold 80/10/10
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg_tag, norm_cfg, pair_cfg):
    cfg_dir = os.path.join(OUTPUT_DIR, cfg_tag)
    relation_means = []

    for rel in RELATIONS:
        pkl_path = os.path.join(cfg_dir, f"HistZigZag_{rel}.pkl")
        if not os.path.exists(pkl_path):
            return None

        with open(pkl_path, "rb") as f:
            feats = np.array(pickle.load(f), dtype=np.float32)

        mat  = sio.loadmat(os.path.join(MAT_DIR, f"LBP_{rel.lower()}.mat"))
        idxa = mat['idxa'].flatten() - 1
        idxb = mat['idxb'].flatten() - 1
        fold = mat['fold'].flatten()
        y    = mat['matches'].flatten()

        X = build_pairs(feats, idxa, idxb, pair_cfg).astype(np.float32)

        fold_scores = []
        for fi in range(1, 6):
            torch.manual_seed(42); np.random.seed(42); random.seed(42)

            fold_mask           = (fold == fi)
            test_mask, val_mask = split_fold_half(fold_mask, seed=42 + fi)
            train_mask          = ~fold_mask

            X_tr = X[train_mask]; X_va = X[val_mask]; X_te = X[test_mask]
            y_tr = y[train_mask]; y_va = y[val_mask]; y_te = y[test_mask]

            X_tr_n, X_va_n = apply_norm(X_tr, X_va, norm_cfg)
            _,      X_te_n = apply_norm(X_tr, X_te, norm_cfg)

            model = train_lcnn(X_tr_n.astype(np.float32), y_tr,
                               X_va_n.astype(np.float32), y_va,
                               input_dim=X_tr_n.shape[1])

            acc = evaluate(model, make_loader(X_te_n.astype(np.float32), y_te))
            fold_scores.append(acc)

        relation_means.append(float(np.mean(fold_scores)))

    return relation_means

# ══════════════════════════════════════════════════════════════════════════════
# ⑧  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if EXTRACT:
        run_extraction()

    print("\n" + "="*70)
    print("  GRID SEARCH — HistZigZag LBP × normalizations × pair fusions (LCNN)")
    print("="*70)

    all_records = []
    n_total = len(EXTRACT_CONFIGS) * len(NORM_CONFIGS) * len(PAIR_CONFIGS)
    count   = 0

    for cfg in EXTRACT_CONFIGS:
        for norm_cfg in NORM_CONFIGS:
            for pair_cfg in PAIR_CONFIGS:
                count += 1
                norm_str = norm_cfg if isinstance(norm_cfg, str) else "+".join(norm_cfg)
                pair_str = pair_cfg if isinstance(pair_cfg, str) else "+".join(pair_cfg)

                result = run_experiment(cfg["tag"], norm_cfg, pair_cfg)
                if result is None:
                    print(f"  [{count:>3}/{n_total}] SKIPPED: {cfg['tag']}")
                    continue

                overall = float(np.mean(result))
                all_records.append({
                    "tag": cfg["tag"], "norm": norm_str, "pair": pair_str,
                    "FD": result[0], "FS": result[1], "MD": result[2], "MS": result[3],
                    "overall": overall,
                })
                print(f"  [{count:>3}/{n_total}] {cfg['tag']:<12} "
                      f"norm={norm_str:<18} pair={pair_str:<40} "
                      f"-> {overall*100:.2f}%  "
                      f"(FD={result[0]*100:.1f} FS={result[1]*100:.1f} "
                      f"MD={result[2]*100:.1f} MS={result[3]*100:.1f})")

    if all_records:
        all_records.sort(key=lambda x: -x["overall"])
        print(f"\n{'='*70}")
        print("  TOP 15 CONFIGURATIONS")
        print(f"{'='*70}")
        print(f"{'#':<3} {'Tag':<14} {'Norm':<20} {'Pair':<42} "
              f"{'FD':>6} {'FS':>6} {'MD':>6} {'MS':>6} {'Overall':>8}")
        print("-"*115)
        for i, r in enumerate(all_records[:15]):
            print(f"{i+1:<3} {r['tag']:<14} {r['norm']:<20} {r['pair']:<42} "
                  f"{r['FD']*100:>5.1f}% {r['FS']*100:>5.1f}% "
                  f"{r['MD']*100:>5.1f}% {r['MS']*100:>5.1f}% "
                  f"{r['overall']*100:>7.2f}%")
        best = all_records[0]
        print(f"\n  Best overall : {best['overall']*100:.2f}%")
        print(f"  Config       : {best['tag']}")
        print(f"  Norm         : {best['norm']}")
        print(f"  Pair fusion  : {best['pair']}")
        print(f"  Friend's     : 81.30% (ps16_ss8, StandardScaler, diff+prod+cos)")
        print(f"  Reference    : 88.00% (HistLBP+SVM)")