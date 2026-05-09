# ================================================================
#  KINSHIP VERIFICATION — Deep Learning Methods Grid Search
#  Models   : ArcFace | FaceNet | ResNet50 | VGGFace | VGG19 | ResNet101
#  Toggle which models, normalizations and fusions to run below
#  Dataset  : KinFaceW-II  |  Protocol : 5-fold CV
# ================================================================

import os, pickle, random, time, warnings
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
MAT_DIR       = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

ARCFACE_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\ArcFace\arcface_embeddings"
FACENET_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\FaceNet\facenet_embeddings"
RESNET50_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\ResNet50\resnet50_embeddings"
VGGFACE_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\VGGFace\vggface_embeddings"
VGG19_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\VGG19\vgg19_embeddings"
RESNET101_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\ResNet101\resnet101_embeddings"

# Maps model name → (folder, file_prefix)
# File expected: <folder>/<prefix>_<REL>.pkl  e.g. ArcFace_FD.pkl
ALL_MODELS = {
    "ArcFace"  : (ARCFACE_DIR,   "ArcFace"),
    "FaceNet"  : (FACENET_DIR,   "FaceNet"),
    "ResNet50" : (RESNET50_DIR,  "ResNet50"),
    "VGGFace"  : (VGGFACE_DIR,   "VGGFace"),
    "VGG19"    : (VGG19_DIR,     "VGG19"),
    "ResNet101": (RESNET101_DIR, "ResNet101"),
}

# ════════════════════════════════════════════════════════════
#  ② TOGGLE — choose what to run
# ════════════════════════════════════════════════════════════

# ── Which models to include ───────────────────────────────────
# Each entry runs that model individually AND adds it to "Combined"
ACTIVE_MODELS = [
    "ArcFace",
    "FaceNet",
    "ResNet50",
    "VGGFace",
    "VGG19",       # uncomment when embeddings are ready
    "ResNet101",   # uncomment when embeddings are ready
]

# ── Run each model individually as well as combined? ─────────
RUN_INDIVIDUAL = True    # True = test each model alone
RUN_COMBINED   = False    # True = test all ACTIVE_MODELS together

# ── Normalizations to test ────────────────────────────────────
NORM_CONFIGS = [
    #"none",
    #"power",
    #"l2",
    #"zscore",
    ("power", "l2"),
    #("power", "zscore"),
    #("l2",    "zscore"),
]

# ── Pair fusions to test ──────────────────────────────────────
PAIR_CONFIGS = [
    #("abs_diff",),
    #("abs_diff", "chi2"),
    #("abs_diff", "product"),
    #("abs_diff", "cosine"),
    #("abs_diff", "chi2",  "product"),
    #("abs_diff", "chi2",  "cosine"),
    #("abs_diff", "product", "cosine"),
    ("abs_diff", "chi2",  "product", "cosine"),
    #("abs_diff", "product", "euclidean", "cosine"),
    #("abs_diff", "chi2",  "product", "euclidean", "cosine"),
]

# ════════════════════════════════════════════════════════════
#  ③ CONFIG
# ════════════════════════════════════════════════════════════
RELATIONS = ["FD", "FS", "MD", "MS"]
REL_NAMES = {
    "FD": "Father-Daughter", "FS": "Father-Son",
    "MD": "Mother-Daughter", "MS": "Mother-Son",
}

REL_HP = {
    "FD": dict(lr=8e-5, weight_decay=2e-3, epochs=800, patience=80,
               mixup_alpha=0.5,  drop=0.50, drop_cls2=0.30,
               se_r=4, pw_boost=1.2, use_sampler=True, batch=16, scheduler="cosine"),
    "FS": dict(lr=3e-4, weight_decay=5e-4, epochs=600, patience=50,
               mixup_alpha=0.4,  drop=0.40, drop_cls2=0.25,
               se_r=4, pw_boost=1.0, use_sampler=True, batch=16, scheduler="cosine"),
    "MD": dict(lr=5e-4, weight_decay=2e-4, epochs=500, patience=50,
               mixup_alpha=0.20, drop=0.25, drop_cls2=0.10,
               se_r=4, pw_boost=1.8, use_sampler=True, batch=16, scheduler="warmrestart"),
    "MS": dict(lr=4e-4, weight_decay=3e-4, epochs=600, patience=70,
               mixup_alpha=0.20, drop=0.25, drop_cls2=0.12,
               se_r=4, pw_boost=1.2, use_sampler=True, batch=16, scheduler="warmrestart"),
}

# ════════════════════════════════════════════════════════════
#  ④ EMBEDDING LOADER
# ════════════════════════════════════════════════════════════
def load_embedding(path):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict) and "features" in data:
        feats = np.array(data["features"], dtype=np.float64)
    elif isinstance(data, dict):
        feats = np.array([data[k] for k in sorted(data.keys())], dtype=np.float64)
    else:
        feats = np.array(data, dtype=np.float64)
    return feats.astype(np.float32)

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

def apply_norm_full(feats, tr_mask, te_mask, norm_cfg):
    """
    Apply normalization to full feature array.
    Scaler fitted on train only; returns full-length array
    so idxa/idxb can index directly.
    """
    norms = [norm_cfg] if isinstance(norm_cfg, str) else list(norm_cfg)
    parts = []
    for n in norms:
        col = np.zeros((len(feats), feats.shape[1]), dtype=np.float32)
        _, col_tr = _apply_single_norm(feats[tr_mask], feats[tr_mask], n)
        _, col_te = _apply_single_norm(feats[tr_mask], feats[te_mask],  n)
        col[tr_mask] = col_tr.astype(np.float32)
        col[te_mask] = col_te.astype(np.float32)
        parts.append(col)
    return np.concatenate(parts, axis=1)

# ════════════════════════════════════════════════════════════
#  ⑥ PAIR FUSION
# ════════════════════════════════════════════════════════════
def _single_pair(a, b, method):
    if method == "abs_diff":  return np.abs(a - b)
    if method == "product":   return a * b
    if method == "chi2":      return (a-b)**2 / (np.abs(a) + np.abs(b) + 1e-8)
    if method == "sq_diff":   return (a - b)**2
    if method == "euclidean": return np.linalg.norm(a-b, axis=1, keepdims=True)
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
    def __init__(self, input_dim, drop=0.40, drop_cls2=0.25, se_r=4):
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
def run_experiment(model_names, norm_cfg, pair_cfg):
    """
    model_names : list of model keys — single model or combined
    Returns [FD_acc, FS_acc, MD_acc, MS_acc]
    """
    rel_accs = []
    for rel in RELATIONS:
        # Load all requested model embeddings
        all_feats = {}
        for name in model_names:
            model_dir, prefix = ALL_MODELS[name]
            path = os.path.join(model_dir, f"{prefix}_{rel}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")
            all_feats[name] = load_embedding(path)

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

            parts_tr, parts_te = [], []
            for name in model_names:
                fn = apply_norm_full(all_feats[name], tr_mask, te_mask, norm_cfg)
                parts_tr.append(build_pairs(fn[idxa[tr_mask]], fn[idxb[tr_mask]], pair_cfg))
                parts_te.append(build_pairs(fn[idxa[te_mask]], fn[idxb[te_mask]], pair_cfg))

            X_tr = np.concatenate(parts_tr, axis=1)
            X_te = np.concatenate(parts_te, axis=1)
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
    t0 = time.time()

    # Build experiment list: individual models + combined
    experiments = {}
    if RUN_INDIVIDUAL:
        for name in ACTIVE_MODELS:
            experiments[name] = [name]
    if RUN_COMBINED and len(ACTIVE_MODELS) > 1:
        label = "+".join(ACTIVE_MODELS)
        experiments[label] = list(ACTIVE_MODELS)

    records = []
    total   = len(experiments) * len(NORM_CONFIGS) * len(PAIR_CONFIGS)
    count   = 0

    print(f"\nGrid search: {len(experiments)} model sets × "
          f"{len(NORM_CONFIGS)} norms × {len(PAIR_CONFIGS)} fusions "
          f"= {total} experiments\n")

    for exp_name, model_names in experiments.items():
        for norm_cfg in NORM_CONFIGS:
            for pair_cfg in PAIR_CONFIGS:
                count   += 1
                norm_str = norm_cfg if isinstance(norm_cfg, str) else "+".join(norm_cfg)
                pair_str = "+".join(pair_cfg)
                print(f"  [{count:>4}/{total}] {exp_name:<25} "
                      f"norm={norm_str:<18} pair={pair_str}")

                try:
                    accs = run_experiment(model_names, norm_cfg, pair_cfg)
                except FileNotFoundError as e:
                    print(f"    SKIPPED: {e}"); continue

                overall = float(np.mean(accs))
                records.append({
                    "models": exp_name, "norm": norm_str, "pair": pair_str,
                    "FD": accs[0], "FS": accs[1], "MD": accs[2], "MS": accs[3],
                    "overall": overall,
                })
                print(f"    -> {overall*100:.2f}%  "
                      f"(FD={accs[0]*100:.1f} FS={accs[1]*100:.1f} "
                      f"MD={accs[2]*100:.1f} MS={accs[3]*100:.1f})")

    if not records:
        print("No results."); return

    records.sort(key=lambda x: -x["overall"])

    W = 120
    print("\n\n" + "=" * W)
    print("  TOP 20 CONFIGURATIONS — Deep Learning Methods")
    print("=" * W)
    print(f"  {'#':<3} {'Models':<25} {'Norm':<18} {'Pair':<40} "
          f"{'FD':>6} {'FS':>6} {'MD':>6} {'MS':>6} {'Overall':>8}")
    print("-" * W)
    for i, r in enumerate(records[:20]):
        print(f"  {i+1:<3} {r['models']:<25} {r['norm']:<18} {r['pair']:<40} "
              f"{r['FD']*100:>5.1f}% {r['FS']*100:>5.1f}% "
              f"{r['MD']*100:>5.1f}% {r['MS']*100:>5.1f}% "
              f"{r['overall']*100:>7.2f}%")

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print("=" * W)


main()
