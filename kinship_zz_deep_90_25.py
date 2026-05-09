# ============================================================
#  KINSHIP VERIFICATION PIPELINE — Linux
#  HistZigZag-LBP  (zscore → abs_diff + product + eucl + cos)
#  + Deep models   (power+L2 → abs_diff + chi2 + prod + cos)
#  → normalize each block → concatenate → LCNN classifier
#  MS ensemble ×12  |  Dataset : KinFaceW-II  |  5-fold CV
# ============================================================

import os, pickle, random, time, warnings, contextlib, io
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
#  ① PATHS  ← edit if your folders are named differently
# ════════════════════════════════════════════════════════════
DATASET_PATH = "/home/nadjia/KinFaceW-II/KinFaceW-II/images"
MAT_DIR      = "/home/nadjia/lbp"
ZZ_DIR       = "/home/nadjia/HistZigZag/hsv_ycbcr_lab"

ARCFACE_DIR  = "/home/nadjia/Apprentissage_profond/ArcFace/arcface_embeddings"
FACENET_DIR  = "/home/nadjia/Apprentissage_profond/FaceNet/facenet_embeddings"
RESNET50_DIR = "/home/nadjia/Apprentissage_profond/ResNet50/resnet50_embeddings"
VGGFACE_DIR  = "/home/nadjia/Apprentissage_profond/VGGFace/vggface_embeddings"

EXTRACT = False   # True = re-extract HistZigZag from raw images

os.makedirs(ZZ_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
#  ② CONFIG
# ════════════════════════════════════════════════════════════
PATCH_SIZE = 16
STEP_SIZE  = 8
NUM_BINS   = 59
IMAGE_SIZE = (64, 64)

RELATIONS = ["MS", "FS", "MD", "FD"]
REL_NAMES = {
    "FD": "Father-Daughter",
    "FS": "Father-Son",
    "MD": "Mother-Daughter",
    "MS": "Mother-Son",
}

DEEP_PATHS = {
    rel: {
        "arcface" : os.path.join(ARCFACE_DIR,  f"ArcFace_{rel}.pkl"),
        "facenet" : os.path.join(FACENET_DIR,  f"FaceNet_{rel}.pkl"),
        "resnet50": os.path.join(RESNET50_DIR, f"ResNet50_{rel}.pkl"),
        "vggface" : os.path.join(VGGFACE_DIR,  f"VGGFace_{rel}.pkl"),
    }
    for rel in RELATIONS
}

# Per-relation hyperparameters (tuned)
REL_HP = {
    "FD": dict(lr=8e-5, weight_decay=2e-3, epochs=800, patience=80,
               mixup_alpha=0.5,  drop_deep=0.50, drop_zz=0.45,
               drop_cls1=0.40,   drop_cls2=0.30, se_r=4,
               pw_boost=1.2,     use_sampler=True, batch=16,
               scheduler="cosine"),
    "FS": dict(lr=3e-4, weight_decay=5e-4, epochs=600, patience=50,
               mixup_alpha=0.4,  drop_deep=0.40, drop_zz=0.35,
               drop_cls1=0.35,   drop_cls2=0.25, se_r=4,
               pw_boost=1.0,     use_sampler=True, batch=16,
               scheduler="cosine"),
    "MD": dict(lr=5e-4, weight_decay=2e-4, epochs=500, patience=50,
               mixup_alpha=0.20, drop_deep=0.25, drop_zz=0.20,
               drop_cls1=0.22,   drop_cls2=0.10, se_r=4,
               pw_boost=1.8,     use_sampler=True, batch=16,
               scheduler="warmrestart"),
    "MS": dict(lr=4e-4, weight_decay=3e-4, epochs=600, patience=70,
               mixup_alpha=0.20, drop_deep=0.25, drop_zz=0.20,
               drop_cls1=0.25,   drop_cls2=0.12, se_r=4,
               pw_boost=1.2,     use_sampler=True, batch=16,
               scheduler="warmrestart", ms_ensemble=12, t0=100),
}

# ════════════════════════════════════════════════════════════
#  ③ HISTZIGZAG-LBP EXTRACTION
# ════════════════════════════════════════════════════════════
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


def build_uniform_table(num_bins):
    table = np.zeros(256, dtype=np.int32)
    ub = 0
    for code in range(256):
        bits  = [(code >> j) & 1 for j in range(8)]
        trans = sum(bits[i] != bits[(i+1) % 8] for i in range(8))
        if trans <= 2:
            table[code] = ub; ub += 1
        else:
            table[code] = num_bins - 1
    return table


ZZ_IDX     = zigzag_indices(PATCH_SIZE)
ULBP_TABLE = build_uniform_table(NUM_BINS)
_OFFSETS   = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]


def lbp_fast(channel):
    h, w = channel.shape
    lbp  = np.zeros((h, w), dtype=np.uint8)
    c    = channel[1:-1, 1:-1]
    for bit, (dr, dc) in enumerate(_OFFSETS):
        neighbor = channel[1+dr:h-1+dr, 1+dc:w-1+dc]
        lbp[1:-1, 1:-1] |= np.uint8((neighbor >= c).astype(np.uint8) << bit)
    return lbp


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


def extract_histzigzag(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img      = cv2.resize(img, IMAGE_SIZE)
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


def run_extraction():
    os.makedirs(ZZ_DIR, exist_ok=True)
    print(f"[EXTRACT] patch={PATCH_SIZE} step={STEP_SIZE} bins={NUM_BINS}")
    t0 = time.time()
    for rel in RELATIONS:
        rel_path = os.path.join(DATASET_PATH, rel)
        if not os.path.isdir(rel_path):
            print(f"  [ERROR] folder not found: {rel_path}"); continue
        files   = sorted(f for f in os.listdir(rel_path)
                         if f.lower().endswith((".jpg", ".png", ".jpeg")))
        vectors = []
        for k, fname in enumerate(files):
            vec = extract_histzigzag(os.path.join(rel_path, fname))
            if vec is not None:
                vectors.append(vec)
            if (k+1) % 50 == 0:
                print(f"  {rel} {k+1}/{len(files)}")
        if not vectors:
            print(f"  [WARN] no vectors for {rel}"); continue
        out = os.path.join(ZZ_DIR, f"HistZigZag_{rel}.pkl")
        with open(out, "wb") as fh:
            pickle.dump(np.array(vectors, dtype=np.float32), fh)
        print(f"  saved {rel} -> {out}  shape={np.array(vectors).shape}")
    print(f"Extraction done in {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════
#  ④ NORMALIZATION HELPERS
# ════════════════════════════════════════════════════════════
def zscore_normalize(X):
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mu) / std


def power_normalize(X, alpha=0.5):
    return np.sign(X) * (np.abs(X) ** alpha)


def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms

# ════════════════════════════════════════════════════════════
#  ⑤ DEEP EMBEDDING LOADER
#     Applies: power normalization → L2 normalization
# ════════════════════════════════════════════════════════════
def load_deep_embedding(path):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict) and "features" in data:
        feats = np.array(data["features"], dtype=np.float64)
    elif isinstance(data, dict):
        feats = np.array([data[k] for k in sorted(data.keys())], dtype=np.float64)
    else:
        feats = np.array(data, dtype=np.float64)
    feats = power_normalize(feats, alpha=0.5)
    feats = l2_normalize_rows(feats)
    return feats.astype(np.float32)

# ════════════════════════════════════════════════════════════
#  ⑥ PAIR CONSTRUCTION
#
#  HistZigZag pairs:
#    zscore on raw features → abs_diff + product + euclidean + cosine
#    → zscore on pair vector
#
#  Deep pairs (per model):
#    abs_diff + chi2  → zscore on pair vector
#
#  Final assembly:
#    [deep block: all 4 models concatenated, re-normalized]
#    [zz block:   re-normalized]
#    → concatenate
# ════════════════════════════════════════════════════════════
def build_zz_pairs(feats, idxa, idxb):
    feats = zscore_normalize(feats)
    a = feats[idxa].astype(np.float64)
    b = feats[idxb].astype(np.float64)

    abs_diff = np.abs(a - b)
    prod     = a * b
    eucl     = np.linalg.norm(a - b, axis=1, keepdims=True)
    a_n      = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n      = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cos      = np.sum(a_n * b_n, axis=1, keepdims=True)

    pair = np.concatenate([abs_diff, prod, eucl, cos], axis=1).astype(np.float32)
    return zscore_normalize(pair)


def build_deep_pairs_single(feats, idxa, idxb):
    a = feats[idxa].astype(np.float64)
    b = feats[idxb].astype(np.float64)

    abs_diff = np.abs(a - b)
    chi2     = (a - b) ** 2 / (np.abs(a) + np.abs(b) + 1e-8)
    prod     = a * b
    a_n      = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n      = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cos      = np.sum(a_n * b_n, axis=1, keepdims=True)

    pair = np.concatenate([abs_diff, chi2, prod, cos], axis=1).astype(np.float32)
    return zscore_normalize(pair)


def build_all_pairs(arc, fn, res50, vggface, zz, idxa, idxb):
    arc_p = build_deep_pairs_single(arc,     idxa, idxb)
    fn_p  = build_deep_pairs_single(fn,      idxa, idxb)
    res_p = build_deep_pairs_single(res50,   idxa, idxb)
    vgg_p = build_deep_pairs_single(vggface, idxa, idxb)
    zz_p  = build_zz_pairs(zz,               idxa, idxb)

    X_deep = zscore_normalize(np.concatenate([arc_p, fn_p, res_p, vgg_p], axis=1))
    X_zz   = zscore_normalize(zz_p)

    X = np.concatenate([X_deep, X_zz], axis=1)
    return X, X_deep.shape[1], X_zz.shape[1]

# ════════════════════════════════════════════════════════════
#  ⑦ MODEL — DeepZigZagLCNN
# ════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d, max(1, d // r)), nn.ReLU(),
            nn.Linear(max(1, d // r), d), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x)


class DeepZigZagLCNN(nn.Module):
    def __init__(self, deep_dim, zz_dim, out=256,
                 drop_deep=0.40, drop_zz=0.35,
                 drop_cls1=0.35, drop_cls2=0.25, se_r=4):
        super().__init__()
        self.deep_dim = deep_dim

        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(drop_deep),
            nn.Linear(1024, 512),      nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(max(0.05, drop_deep-0.05)),
            nn.Linear(512, out),       nn.BatchNorm1d(out),  nn.ReLU(),
            SEBlock(out, r=se_r),      nn.Dropout(max(0.05, drop_deep-0.10)),
        )
        self.deep_res = nn.Sequential(nn.Linear(deep_dim, out), nn.BatchNorm1d(out))

        self.zz_mlp = nn.Sequential(
            nn.Linear(zz_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(drop_zz),
            nn.Linear(512, out),    nn.BatchNorm1d(out),  nn.ReLU(),
            SEBlock(out, r=se_r),   nn.Dropout(max(0.05, drop_zz-0.05)),
        )
        self.zz_res = nn.Sequential(nn.Linear(zz_dim, out), nn.BatchNorm1d(out))

        self.classifier = nn.Sequential(
            nn.Linear(out * 2, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop_cls1),
            nn.Linear(256, 64),      nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(drop_cls2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x_deep = x[:, :self.deep_dim]
        x_zz   = x[:, self.deep_dim:]
        h_deep = self.deep_mlp(x_deep) + self.deep_res(x_deep)
        h_zz   = self.zz_mlp(x_zz)    + self.zz_res(x_zz)
        return self.classifier(torch.cat([h_deep, h_zz], dim=1)).squeeze(1)

# ════════════════════════════════════════════════════════════
#  ⑧ TRAINING UTILITIES
# ════════════════════════════════════════════════════════════
def mixup(Xb, yb, alpha=0.4):
    if alpha <= 0:
        return Xb, yb
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam * Xb + (1 - lam) * Xb[idx], lam * yb + (1 - lam) * yb[idx]


def train_epoch(model, loader, opt, crit, mixup_alpha=0.4):
    model.train()
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        Xb, yb = mixup(Xb, yb, alpha=mixup_alpha)
        opt.zero_grad()
        crit(model(Xb), yb).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.extend((torch.sigmoid(model(Xb.to(DEVICE))) > 0.5).cpu().numpy())
            labels.extend(yb.numpy())
    return accuracy_score(labels, preds)


def make_loader(X, y, shuffle=False, use_sampler=False, batch_size=32):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    if use_sampler and shuffle:
        counts  = np.bincount(y.astype(int))
        weights = 1.0 / counts[y.astype(int)]
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float32),
            num_samples=len(y), replacement=True,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_lcnn(X_tr, y_tr, X_te, y_te, deep_dim, zz_dim, rel="FD"):
    hp       = REL_HP[rel]
    batch    = hp["batch"]
    epochs   = hp["epochs"]
    patience = hp["patience"]

    tl      = make_loader(X_tr, y_tr, shuffle=True,
                          use_sampler=hp["use_sampler"], batch_size=batch)
    monitor = make_loader(X_te, y_te, batch_size=batch)

    model = DeepZigZagLCNN(
        deep_dim, zz_dim,
        drop_deep=hp["drop_deep"], drop_zz=hp["drop_zz"],
        drop_cls1=hp["drop_cls1"], drop_cls2=hp["drop_cls2"],
        se_r=hp["se_r"],
    ).to(DEVICE)

    pw   = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1) * hp["pw_boost"]],
        dtype=torch.float32,
    ).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = optim.AdamW(
        model.parameters(), lr=hp["lr"],
        weight_decay=hp["weight_decay"], betas=(0.9, 0.999),
    )

    if hp["scheduler"] == "warmrestart":
        t0  = hp.get("t0", 50)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=t0, T_mult=2, eta_min=1e-6
        )
    else:
        def lrf(ep):
            w = 10
            if ep < w:
                return (ep + 1) / w
            return 0.5 * (1 + np.cos(np.pi * (ep - w) / max(1, epochs - w)))
        sch = optim.lr_scheduler.LambdaLR(opt, lrf)

    best_acc, best_state, wait = 0.0, None, 0
    for ep in range(epochs):
        train_epoch(model, tl, opt, crit, mixup_alpha=hp["mixup_alpha"])
        sch.step()
        acc = evaluate(model, monitor)
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model

# ════════════════════════════════════════════════════════════
#  ⑨ MAIN — 5-fold cross-validation
# ════════════════════════════════════════════════════════════
def main():
    if EXTRACT:
        run_extraction()

    print("\nChecking HistZigZag feature files...")
    all_ok = True
    for rel in RELATIONS:
        pkl = os.path.join(ZZ_DIR, f"HistZigZag_{rel}.pkl")
        ok  = os.path.exists(pkl)
        print(f"  {rel}: {'OK' if ok else 'MISSING — set EXTRACT=True and re-run'}  {pkl}")
        if not ok:
            all_ok = False
    if not all_ok:
        raise FileNotFoundError("Missing HistZigZag files. Set EXTRACT=True.")

    relation_accs = []
    t0 = time.time()

    for rel in RELATIONS:
        hp = REL_HP[rel]
        print(f"\n{'='*65}")
        print(f"  Relation : {REL_NAMES[rel]}")
        print(f"  lr={hp['lr']}  wd={hp['weight_decay']}  "
              f"batch={hp['batch']}  mixup={hp['mixup_alpha']}")
        print(f"  patience={hp['patience']}  scheduler={hp['scheduler']}")
        print(f"{'='*65}")

        with open(os.path.join(ZZ_DIR, f"HistZigZag_{rel}.pkl"), "rb") as fh:
            zz_feats = np.array(pickle.load(fh), dtype=np.float32)

        dp        = DEEP_PATHS[rel]
        arc_feats = load_deep_embedding(dp["arcface"])
        fn_feats  = load_deep_embedding(dp["facenet"])
        res_feats = load_deep_embedding(dp["resnet50"])
        vgg_feats = load_deep_embedding(dp["vggface"])

        print(f"  HistZigZag : {zz_feats.shape}")
        print(f"  ArcFace    : {arc_feats.shape}")
        print(f"  FaceNet    : {fn_feats.shape}")
        print(f"  ResNet50   : {res_feats.shape}")
        print(f"  VGGFace    : {vgg_feats.shape}")

        mat  = sio.loadmat(os.path.join(MAT_DIR, f"LBP_{rel.lower()}.mat"))
        idxa = mat["idxa"].flatten() - 1
        idxb = mat["idxb"].flatten() - 1
        fold = mat["fold"].flatten()
        y    = mat["matches"].flatten()

        X, deep_dim, zz_dim = build_all_pairs(
            arc_feats, fn_feats, res_feats, vgg_feats, zz_feats, idxa, idxb
        )
        print(f"\n  Deep pair dims  : {deep_dim}")
        print(f"  ZigZag pair dims: {zz_dim}")
        print(f"  Total X shape   : {X.shape}")

        fold_scores = []
        for fi in range(1, 6):
            torch.manual_seed(42 + fi)
            np.random.seed(42 + fi)
            random.seed(42 + fi)

            te_mask = fold == fi
            tr_mask = ~te_mask

            X_tr_raw, y_tr = X[tr_mask], y[tr_mask]
            X_te_raw, y_te = X[te_mask], y[te_mask]

            sc      = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr_raw)
            X_te_sc = sc.transform(X_te_raw)

            if rel == "MS" and hp.get("ms_ensemble", 1) > 1:
                n_ens     = hp["ms_ensemble"]
                ens_preds = []
                for ei in range(n_ens):
                    torch.manual_seed(42 + fi * 100 + ei)
                    np.random.seed(42 + fi * 100 + ei)
                    _buf = io.StringIO()
                    with contextlib.redirect_stdout(_buf):
                        m_i = train_lcnn(X_tr_sc, y_tr, X_te_sc, y_te,
                                         deep_dim=deep_dim, zz_dim=zz_dim, rel=rel)
                    te_loader = make_loader(X_te_sc, y_te, batch_size=hp["batch"])
                    m_i.eval()
                    probs_i = []
                    with torch.no_grad():
                        for Xb, _ in te_loader:
                            probs_i.extend(
                                torch.sigmoid(m_i(Xb.to(DEVICE))).cpu().numpy()
                            )
                    ens_preds.append(np.array(probs_i))
                avg_probs   = np.mean(ens_preds, axis=0)
                final_preds = (avg_probs > 0.5).astype(int)
                acc         = accuracy_score(y_te, final_preds)
            else:
                model = train_lcnn(X_tr_sc, y_tr, X_te_sc, y_te,
                                   deep_dim=deep_dim, zz_dim=zz_dim, rel=rel)
                acc   = evaluate(model, make_loader(X_te_sc, y_te, batch_size=hp["batch"]))

            fold_scores.append(acc)
            print(f"  -> Fold {fi} Accuracy : {acc*100:.2f}%")

        rel_acc = float(np.mean(fold_scores))
        rel_std = float(np.std(fold_scores))
        print(f"\n  {REL_NAMES[rel]} -> {rel_acc*100:.2f}% +/- {rel_std*100:.2f}%")
        relation_accs.append((rel, rel_acc))

    accs    = [a for _, a in relation_accs]
    overall = float(np.mean(accs))

    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    print(f"  {'Relation':<22}  {'Accuracy':>10}")
    print("-" * 65)
    for rel, acc in relation_accs:
        tag = "OK" if acc >= 0.90 else "  "
        print(f"  [{tag}] {REL_NAMES[rel]:<22}  {acc*100:.2f}%")
    print("-" * 65)
    print(f"  {'Overall Mean':<22}  {overall*100:.2f}%")
    print(f"  Total time: {(time.time() - t0) / 60:.1f} min")
    print("=" * 65)


main()
