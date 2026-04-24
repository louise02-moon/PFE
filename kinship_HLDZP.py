"""
HistLDZP → LCNN  —  Fixed version matching the 81.3% architecture
5-fold (80/10/10 split) · smaller LCNN · proper pair fusion
"""

import cv2, os, pickle, random, time
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# ①  SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
OUTPUT_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes_classiques\Hist-LDZP\ps16_ss8_k3"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

EXTRACT    = False       # False if features already extracted
PATCH_SIZE = 16
STEP_SIZE  = 8
TOP_K      = 3
NUM_BINS   = 59

EPOCHS     = 600
PATIENCE   = 25
LR         = 5e-4
BATCH      = 32

RELATIONS  = ["FD", "FS", "MD", "MS"]
REL_NAMES  = {"FD": "Father-Daughter", "FS": "Father-Son",
              "MD": "Mother-Daughter", "MS": "Mother-Son"}

# ══════════════════════════════════════════════════════════════════════════════
# ②  SPLIT HELPER — 80/10/10 using KinFaceW-II folds
# ══════════════════════════════════════════════════════════════════════════════

def split_fold_half(fold_mask, seed=42):
    """Split one fold (20%) into two halves: test (10%) + val (10%)."""
    indices = np.where(fold_mask)[0]
    rng     = np.random.RandomState(seed)
    rng.shuffle(indices)
    mid     = len(indices) // 2
    mask_te = np.zeros(len(fold_mask), dtype=bool)
    mask_va = np.zeros(len(fold_mask), dtype=bool)
    mask_te[indices[:mid]] = True
    mask_va[indices[mid:]] = True
    return mask_te, mask_va

# ══════════════════════════════════════════════════════════════════════════════
# ③  KIRSCH + EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

KIRSCH_MASKS = np.array([
    [[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]],
    [[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]],
    [[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]],
    [[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]],
    [[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]],
    [[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]],
    [[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]],
    [[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]],
], dtype=np.float32)

def _build_uniform_table():
    table = np.zeros(256, dtype=np.int32)
    uid = 0
    for code in range(256):
        bits = [(code >> j) & 1 for j in range(8)]
        t = sum(bits[i] != bits[(i+1) % 8] for i in range(8))
        if t <= 2: table[code] = uid; uid += 1
        else: table[code] = NUM_BINS - 1
    return table

UNIFORM_TABLE = _build_uniform_table()

def compute_ldzp_map(channel):
    ch = channel.astype(np.float32)
    resp = np.zeros((8, *ch.shape), dtype=np.float32)
    for d, mask in enumerate(KIRSCH_MASKS):
        resp[d] = cv2.filter2D(ch, -1, mask, borderType=cv2.BORDER_REFLECT)
    ranked = np.argsort(-np.abs(resp), axis=0)
    lmap = np.zeros(ch.shape, dtype=np.uint8)
    for k in range(TOP_K):
        for d in range(8):
            lmap[ranked[k] == d] += np.uint8(1 << d)
    return lmap

def extract_histldzp(image_path):
    img = cv2.imread(image_path)
    if img is None: return None

    img = cv2.resize(img, (64, 64))
    channels = [
        img[:,:,2], img[:,:,1], img[:,:,0],                           # R G B
        *cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)),              # H S V
        *cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)),            # Y Cr Cb
    ]

    h, w = channels[0].shape
    lmaps = [UNIFORM_TABLE[compute_ldzp_map(ch.astype(np.float64))
             .astype(np.int32)].astype(np.int32) for ch in channels]

    feats = []
    for i in range(0, h - PATCH_SIZE + 1, STEP_SIZE):
        for j in range(0, w - PATCH_SIZE + 1, STEP_SIZE):
            pf = []
            for lm in lmaps:
                patch = lm[i:i+PATCH_SIZE, j:j+PATCH_SIZE].ravel()
                hist, _ = np.histogram(patch, bins=NUM_BINS, range=(0, NUM_BINS))
                hist = hist.astype(np.float32)
                s = hist.sum()
                if s > 0: hist /= s
                pf.append(hist)
            feats.append(np.concatenate(pf))
    return np.concatenate(feats) if feats else None

def run_extraction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nExtraction: patch={PATCH_SIZE} step={STEP_SIZE} top_k={TOP_K} bins={NUM_BINS}")
    t0 = time.time()
    for rel in RELATIONS:
        rel_path = os.path.join(DATASET_PATH, rel)
        files = sorted([f for f in os.listdir(rel_path)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        vecs = []
        print(f"  {rel} — {len(files)} images")
        for k, fn in enumerate(files):
            v = extract_histldzp(os.path.join(rel_path, fn))
            if v is not None: vecs.append(v)
            if (k+1) % 100 == 0: print(f"    {k+1}/{len(files)}")
        vecs = np.array(vecs, dtype=np.float32)
        with open(os.path.join(OUTPUT_DIR, f"HistLDZP_{rel}.pkl"), "wb") as f:
            pickle.dump(vecs, f)
        print(f"    Saved: {vecs.shape}")
    print(f"\nExtraction done in {time.time()-t0:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
# ④  PAIR FUSION — same as the 81.3% code (power norm built-in)
# ══════════════════════════════════════════════════════════════════════════════

def build_pair_features(feats, idxa, idxb):
    a = feats[idxa].astype(np.float64)
    b = feats[idxb].astype(np.float64)

    diff = np.abs(a - b)
    diff = np.sqrt(diff) - np.sqrt(diff).mean(axis=0)   # power + centering

    prod = a * b
    prod = np.sqrt(np.abs(prod)) * np.sign(prod)         # power

    an  = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn  = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cos = np.sum(an * bn, axis=1, keepdims=True)

    return np.concatenate([diff, prod, cos], axis=1).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# ⑤  NORMALIZATION — StandardScaler fit on train only
# ══════════════════════════════════════════════════════════════════════════════

def normalize_splits(X_tr, X_va, X_te):
    sc = StandardScaler()
    return (sc.fit_transform(X_tr).astype(np.float32),
            sc.transform(X_va).astype(np.float32),
            sc.transform(X_te).astype(np.float32))

# ══════════════════════════════════════════════════════════════════════════════
# ⑥  LCNN — SMALLER architecture matching the 81.3% code
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d, max(1, d//r)), nn.ReLU(),
            nn.Linear(max(1, d//r), d), nn.Sigmoid())
    def forward(self, x): return x * self.se(x)

class LCNN(nn.Module):
    """Smaller LCNN — matches the architecture that gave 81.3%"""
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
    return lam*Xb + (1-lam)*Xb[idx], lam*yb + (1-lam)*yb[idx]

def train_epoch(model, loader, opt, crit):
    model.train()
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        Xb, yb = mixup(Xb, yb)
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

def make_loader(X, y, shuffle=False):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=BATCH, shuffle=shuffle)

def train_lcnn(X_tr, y_tr, X_va, y_va, input_dim):
    tl    = make_loader(X_tr, y_tr, shuffle=True)
    vl    = make_loader(X_va, y_va)
    model = LCNN(input_dim).to(DEVICE)

    pw   = torch.tensor([(y_tr==0).sum()/max((y_tr==1).sum(),1)],
                        dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4, betas=(0.9, 0.999))

    def lrf(ep):
        warmup = 10
        if ep < warmup: return (ep+1)/warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(1, EPOCHS - warmup)))

    sch = optim.lr_scheduler.LambdaLR(opt, lrf)
    best_acc, best_state, wait = 0.0, None, 0

    for ep in range(EPOCHS):
        train_epoch(model, tl, opt, crit)
        sch.step()
        acc = evaluate(model, vl)
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop ep {ep+1} | best val={best_acc*100:.2f}%")
                break

    model.load_state_dict(best_state)
    return model

# ══════════════════════════════════════════════════════════════════════════════
# ⑦  MAIN — 5-fold (80/10/10)
# ══════════════════════════════════════════════════════════════════════════════

if EXTRACT:
    run_extraction()

all_results = {}

for rel in RELATIONS:
    print(f"\n{'='*60}")
    print(f"  {REL_NAMES[rel]}")
    print(f"{'='*60}")

    pkl = os.path.join(OUTPUT_DIR, f"HistLDZP_{rel}.pkl")
    if not os.path.exists(pkl):
        print(f"  MISSING: {pkl}"); continue

    with open(pkl, "rb") as f:
        feats = np.array(pickle.load(f), dtype=np.float32)
    print(f"  Features: {feats.shape}")

    mat  = sio.loadmat(os.path.join(MAT_DIR, f"LBP_{rel.lower()}.mat"))
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    X = build_pair_features(feats, idxa, idxb)
    print(f"  Pairs: {X.shape}")

    fold_scores = []
    for fi in range(1, 6):
        torch.manual_seed(42); np.random.seed(42); random.seed(42)

        # 80/10/10 split using KinFaceW-II folds
        fold_mask           = (fold == fi)
        test_mask, val_mask = split_fold_half(fold_mask, seed=42 + fi)
        train_mask          = ~fold_mask

        X_tr_raw = X[train_mask]
        X_va_raw = X[val_mask]
        X_te_raw = X[test_mask]
        y_tr     = y[train_mask]
        y_va     = y[val_mask]
        y_te     = y[test_mask]

        n = len(y)
        print(f"  Fold {fi} → train={len(y_tr)} ({100*len(y_tr)/n:.0f}%)  "
              f"val={len(y_va)} ({100*len(y_va)/n:.0f}%)  "
              f"test={len(y_te)} ({100*len(y_te)/n:.0f}%)")

        X_tr_sc, X_va_sc, X_te_sc = normalize_splits(X_tr_raw, X_va_raw, X_te_raw)

        model = train_lcnn(X_tr_sc, y_tr, X_va_sc, y_va, input_dim=X_tr_sc.shape[1])

        acc = evaluate(model, make_loader(X_te_sc, y_te))
        fold_scores.append(acc)
        print(f"    → Test: {acc*100:.2f}%")

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))
    all_results[rel] = {"scores": fold_scores, "mean": mean_acc, "std": std_acc}
    print(f"  {REL_NAMES[rel]}: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  FINAL SUMMARY — HistLDZP → LCNN (5-fold 80/10/10)")
print(f"{'='*60}")
for rel, res in all_results.items():
    print(f"  {REL_NAMES[rel]:<22} {res['mean']*100:.2f}% ± {res['std']*100:.2f}%")
    print(f"  {'':22} folds: {[f'{s*100:.1f}%' for s in res['scores']]}")
overall = float(np.mean([r["mean"] for r in all_results.values()]))
print(f"\n  Overall  : {overall*100:.2f}%")
print(f"  Friend's : 81.30% (HistZigZag)")