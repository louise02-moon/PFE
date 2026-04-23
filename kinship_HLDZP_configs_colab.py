import cv2, os, pickle, random, itertools
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
# ①  USER SETTINGS — edit only this section
# ══════════════════════════════════════════════════════════════════════════════

DRIVE_ROOT   = "/content/drive/MyDrive"
DATASET_PATH = f"{DRIVE_ROOT}/KinFaceW-II/KinFaceW-II/images"
OUTPUT_DIR   = f"{DRIVE_ROOT}/Methodes_classiques/Hist-LDZP"
MAT_DIR      = "/content/PFE/lbp"

EXTRACT = False   # True = extract features first, False = skip to LCNN grid

# ── Extraction grid ───────────────────────────────────────────────────────────
EXTRACT_CONFIGS = [
    {"patch_size": 32, "step_size": 2, "top_k": 3, "num_bins": 59, "tag": "ps32_ss2_k3"},
    {"patch_size": 16, "step_size": 2, "top_k": 3, "num_bins": 59, "tag": "ps16_ss2_k3"},
    {"patch_size": 16, "step_size": 2, "top_k": 4, "num_bins": 59, "tag": "ps16_ss2_k4"},
    {"patch_size": 16, "step_size": 4, "top_k": 3, "num_bins": 59, "tag": "ps16_ss4_k3"},
    {"patch_size": 32, "step_size": 4, "top_k": 3, "num_bins": 59, "tag": "ps32_ss4_k3"},
]

# ── Normalizations to test ────────────────────────────────────────────────────
# str = single norm, tuple = apply each independently then concatenate
NORM_CONFIGS = [
    #"power",               # sqrt(|x|) then mean-center — best for histograms
    #"l1",                  # divide by L1 norm
    "l2",                  # unit vector
    #"zscore",              # zero mean unit variance
    #("power", "l2"),       # both concatenated
    #("power", "zscore"),   # both concatenated
]

# ── Pair fusion methods to test ───────────────────────────────────────────────
# str = single method, tuple = concatenate multiple
PAIR_CONFIGS = [
    "abs_diff",
    #("abs_diff", "product"),
    #("abs_diff", "product", "cosine"),
    #("abs_diff", "product", "euclidean", "cosine"),
]

# ── LCNN training settings ────────────────────────────────────────────────────
LCNN_EPOCHS   = 300
LCNN_PATIENCE = 25
LCNN_LR       = 5e-4
LCNN_BATCH    = 32

RELATIONS = ["FD", "FS", "MD", "MS"]
REL_NAMES = {"FD": "Father-Daughter", "FS": "Father-Son",
             "MD": "Mother-Daughter", "MS": "Mother-Son"}

# ══════════════════════════════════════════════════════════════════════════════
# ②  KIRSCH + UNIFORM TABLE
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

def _build_uniform_table(num_bins):
    table = np.zeros(256, dtype=np.int32)
    uniform_bin = 0
    for code in range(256):
        bits = [(code >> j) & 1 for j in range(8)]
        transitions = sum(bits[i] != bits[(i+1) % 8] for i in range(8))
        if transitions <= 2:
            table[code] = uniform_bin; uniform_bin += 1
        else:
            table[code] = num_bins - 1
    return table

# ══════════════════════════════════════════════════════════════════════════════
# ③  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ldzp_map(channel, top_k):
    ch = channel.astype(np.float32)
    responses = np.zeros((8, *ch.shape), dtype=np.float32)
    for d, mask in enumerate(KIRSCH_MASKS):
        responses[d] = cv2.filter2D(ch, -1, mask, borderType=cv2.BORDER_REFLECT)
    ranked   = np.argsort(-np.abs(responses), axis=0)
    ldzp_map = np.zeros(ch.shape, dtype=np.uint8)
    for k in range(top_k):
        for d in range(8):
            ldzp_map[ranked[k] == d] += np.uint8(1 << d)
    return ldzp_map

def extract_histldzp(image_path, patch_size, step_size, top_k, num_bins):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return None

    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    R, G, B   = img_rgb[:,:,0],   img_rgb[:,:,1],   img_rgb[:,:,2]
    H, S, V   = img_hsv[:,:,0],   img_hsv[:,:,1],   img_hsv[:,:,2]
    Y, Cr, Cb = img_ycbcr[:,:,0], img_ycbcr[:,:,1], img_ycbcr[:,:,2]
    channels  = [R, G, B, H, S, V, Y, Cb, Cr]

    table    = _build_uniform_table(num_bins)
    h, w     = R.shape
    ldzp_maps = [table[compute_ldzp_map(ch.astype(np.float64), top_k)
                       .astype(np.int32)].astype(np.int32) for ch in channels]

    feature_list = []
    for i in range(0, h - patch_size + 1, step_size):
        for j in range(0, w - patch_size + 1, step_size):
            patch_feat = []
            for lmap in ldzp_maps:
                patch = lmap[i:i+patch_size, j:j+patch_size].ravel()
                hist, _ = np.histogram(patch, bins=num_bins, range=(0, num_bins))
                hist = hist.astype(np.float32)
                s = hist.sum()
                if s > 0: hist /= s
                patch_feat.append(hist)
            feature_list.append(np.concatenate(patch_feat))
    return np.concatenate(feature_list) if feature_list else None

def run_extraction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cfg in EXTRACT_CONFIGS:
        tag     = cfg["tag"]
        cfg_dir = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"\nExtracting: {tag}")
        for rel in RELATIONS:
            rel_path = os.path.join(DATASET_PATH, rel)
            files    = sorted([f for f in os.listdir(rel_path)
                               if f.lower().endswith((".jpg",".png",".jpeg"))])
            vectors  = []
            for k, fname in enumerate(files):
                vec = extract_histldzp(os.path.join(rel_path, fname),
                                       cfg["patch_size"], cfg["step_size"],
                                       cfg["top_k"], cfg["num_bins"])
                if vec is not None: vectors.append(vec)
                if (k+1) % 50 == 0: print(f"  {rel} {k+1}/{len(files)}")
            vectors = np.array(vectors, dtype=np.float32)
            with open(os.path.join(cfg_dir, f"HistLDZP_{rel}.pkl"), "wb") as f:
                pickle.dump(vectors, f)
            print(f"  {rel}: {vectors.shape}")
    print("Extraction complete.")

# ══════════════════════════════════════════════════════════════════════════════
# ④  NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _single_norm_tr_te(Xtr, Xte, name):
    """Apply one norm fitted on train, applied to test."""
    if name == "power":
        Xtr_n = np.sqrt(np.abs(Xtr)); mu = Xtr_n.mean(0)
        return Xtr_n - mu, np.sqrt(np.abs(Xte)) - mu
    if name == "l1":
        s = np.abs(Xtr).sum(1, keepdims=True) + 1e-8
        st = np.abs(Xte).sum(1, keepdims=True) + 1e-8
        return Xtr / s, Xte / st
    if name == "l2":
        return sk_normalize(Xtr, "l2"), sk_normalize(Xte, "l2")
    if name == "zscore":
        sc = StandardScaler(); Xtr_n = sc.fit_transform(Xtr)
        return Xtr_n, sc.transform(Xte)
    if name == "minmax":
        sc = MinMaxScaler(); Xtr_n = sc.fit_transform(Xtr)
        return Xtr_n, sc.transform(Xte)
    if name == "none":
        return Xtr, Xte
    raise ValueError(f"Unknown norm: {name}")

def apply_norm(Xtr, Xte, norm_cfg):
    """Single norm or tuple → concatenate results."""
    if isinstance(norm_cfg, str):
        return _single_norm_tr_te(Xtr, Xte, norm_cfg)
    parts_tr, parts_te = [], []
    for n in norm_cfg:
        ntr, nte = _single_norm_tr_te(Xtr, Xte, n)
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
        an = a / (np.linalg.norm(a, 1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, 1, keepdims=True) + 1e-8)
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
    """Single-branch LCNN for standalone HistLDZP."""
    def __init__(self, input_dim, local_out=256):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.ReLU(), SEBlock(512), nn.Dropout(0.25),
            nn.Linear(512, local_out),  nn.BatchNorm1d(local_out), nn.ReLU(), SEBlock(local_out), nn.Dropout(0.20),
        )
        self.res  = nn.Sequential(nn.Linear(local_out, 128), nn.BatchNorm1d(128))
        self.head = nn.Sequential(
            nn.Linear(local_out, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.branch(x)
        r = self.res(h)
        h = self.head[0](h); h = self.head[1](h); h = self.head[2](h); h = self.head[3](h)
        h = self.head[4](h); h = self.head[5](h); h = self.head[6](h) + r; h = self.head[7](h)
        return self.head[8](h).squeeze(1)

def mixup(Xb, yb, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam*Xb+(1-lam)*Xb[idx], lam*yb+(1-lam)*yb[idx]

def train_ep(m, ld, opt, crit):
    m.train()
    for Xb, yb in ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        Xb, yb = mixup(Xb, yb)
        opt.zero_grad()
        loss = crit(m(Xb), yb); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()

def evaluate(m, ld):
    m.eval(); p, l = [], []
    with torch.no_grad():
        for Xb, yb in ld:
            p.extend((torch.sigmoid(m(Xb.to(DEVICE))) > 0.5).cpu().numpy())
            l.extend(yb.numpy())
    return accuracy_score(l, p)

def train_lcnn(Xtr, ytr, Xva, yva, input_dim,
               epochs=LCNN_EPOCHS, batch=LCNN_BATCH,
               lr=LCNN_LR, patience=LCNN_PATIENCE):
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)
    tl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)
    vl = DataLoader(TensorDataset(Xv, yv), batch_size=batch)

    m  = LCNN(input_dim).to(DEVICE)
    pw = torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],
                      dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = optim.AdamW(m.parameters(), lr=lr, weight_decay=2e-4)

    def lrf(ep):
        return (ep+1)/10 if ep < 10 else \
               0.5*(1+np.cos(np.pi*(ep-10)/max(1,epochs-10)))
    sch = optim.lr_scheduler.LambdaLR(opt, lrf)

    best, state, wait = 0.0, None, 0
    for ep in range(epochs):
        train_ep(m, tl, opt, crit); sch.step()
        acc = evaluate(m, vl)
        if acc > best:
            best  = acc
            state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            wait  = 0
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(state)
    return m

# ══════════════════════════════════════════════════════════════════════════════
# ⑦  RUN ONE EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg_tag, norm_cfg, pair_cfg):
    cfg_dir = os.path.join(OUTPUT_DIR, cfg_tag)
    relation_means = []

    for rel in RELATIONS:
        pkl_path = os.path.join(cfg_dir, f"HistLDZP_{rel}.pkl")
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
        for f in range(1, 6):
            torch.manual_seed(42); np.random.seed(42); random.seed(42)

            tr, te   = fold != f, fold == f
            Xtr_raw  = X[tr]; Xte_raw = X[te]
            y_tr     = y[tr]; y_te    = y[te]

            # Normalize — fit on train, apply to test
            Xtr_sc, Xte_sc = apply_norm(Xtr_raw, Xte_raw, norm_cfg)

            # Inner validation fold for early stopping
            fold_ids = fold[tr]
            val_fold = f % 5 + 1
            vm       = fold_ids == val_fold

            model = train_lcnn(
                Xtr_sc[~vm], y_tr[~vm],
                Xtr_sc[ vm], y_tr[ vm],
                input_dim=Xtr_sc.shape[1]
            )

            te_loader = DataLoader(
                TensorDataset(torch.tensor(Xte_sc, dtype=torch.float32),
                              torch.tensor(y_te,   dtype=torch.float32)),
                batch_size=LCNN_BATCH)
            fold_scores.append(evaluate(model, te_loader))

        relation_means.append(float(np.mean(fold_scores)))

    return relation_means

# ══════════════════════════════════════════════════════════════════════════════
# ⑧  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if EXTRACT:
    run_extraction()

print("\n" + "="*70)
print("  GRID SEARCH — HistLDZP configs × normalizations × pair fusions (LCNN)")
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
                print(f"  [{count:>3}/{n_total}] SKIPPED (files not found): {cfg['tag']}")
                continue

            overall = float(np.mean(result))
            all_records.append({
                "tag": cfg["tag"], "norm": norm_str, "pair": pair_str,
                "FD": result[0], "FS": result[1], "MD": result[2], "MS": result[3],
                "overall": overall,
            })
            print(f"  [{count:>3}/{n_total}] {cfg['tag']:<18} "
                  f"norm={norm_str:<22} pair={pair_str:<38} "
                  f"→ {overall*100:.2f}%  "
                  f"(FD={result[0]*100:.1f} FS={result[1]*100:.1f} "
                  f"MD={result[2]*100:.1f} MS={result[3]*100:.1f})")

# ── Ranked summary ─────────────────────────────────────────────────────────────
if all_records:
    all_records.sort(key=lambda x: -x["overall"])
    print(f"\n{'='*70}")
    print("  TOP 10 CONFIGURATIONS")
    print(f"{'='*70}")
    print(f"{'#':<3} {'Tag':<20} {'Norm':<24} {'Pair':<38} "
          f"{'FD':>6} {'FS':>6} {'MD':>6} {'MS':>6} {'Overall':>8}")
    print("-"*115)
    for i, r in enumerate(all_records[:10]):
        print(f"{i+1:<3} {r['tag']:<20} {r['norm']:<24} {r['pair']:<38} "
              f"{r['FD']*100:>5.1f}% {r['FS']*100:>5.1f}% "
              f"{r['MD']*100:>5.1f}% {r['MS']*100:>5.1f}% "
              f"{r['overall']*100:>7.2f}%")
    best = all_records[0]
    print(f"\n  Best overall : {best['overall']*100:.2f}%")
    print(f"  Config       : {best['tag']}")
    print(f"  Norm         : {best['norm']}")
    print(f"  Pair fusion  : {best['pair']}")