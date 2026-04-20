
"""THE FIX
────────
Replace the SVM entirely with 3 direct similarity scores computed from the
LDZP histograms. These are parameter-free, always well-conditioned, and are
actually the standard metrics used in the LBP/LDP literature:

  1. Cosine similarity      — global orientation match
  2. L2 distance            — global magnitude difference
  3. Chi-squared distance   — THE standard metric for histogram comparison,
                              specifically designed for this kind of descriptor

These 3 numbers replace the unstable SVM score. The LCNN then has:
  [ArcFace pairs, FaceNet pairs, ResNet50 pairs, VGGFace pairs, 3 LDZP scores]

No compression, no leakage risk, no dimensionality problem.
────────────────────────────────────────────────────────────────────────────────
"""

import os, pickle, random
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── Paths ────────────────────────────────────────────────────────────────────
ARCFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
FACENET_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\FaceNet\facenet_embeddings"
RESNET50_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ResNet50\resnet50_embeddings"
VGGFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface_embeddings"
HISTLDZP_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LDZP\HLDZP_feature_vectors"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FD.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_FD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FD.pkl",
        "histldzp": f"{HISTLDZP_DIR}\\HistLDZP_FD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FS.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_FS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FS.pkl",
        "histldzp": f"{HISTLDZP_DIR}\\HistLDZP_FS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MD.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_MD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MD.pkl",
        "histldzp": f"{HISTLDZP_DIR}\\HistLDZP_MD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MS.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_MS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MS.pkl",
        "histldzp": f"{HISTLDZP_DIR}\\HistLDZP_MS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_ms.mat",
    },
}


# ─── Loaders ──────────────────────────────────────────────────────────────────
def load_emb(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "features" in data:
        return np.array(data["features"], dtype=np.float32)
    elif isinstance(data, dict):
        return np.array([data[k] for k in sorted(data, key=lambda k: os.path.basename(k))],
                        dtype=np.float32)
    return np.array(data, dtype=np.float32)


# ─── Deep pair features ───────────────────────────────────────────────────────
def pnorm(X: np.ndarray, a: float = 0.5) -> np.ndarray:
    return np.sign(X) * (np.abs(X) ** a)


def deep_pair(feats: np.ndarray,
              ia: np.ndarray,
              ib: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings, then build 3-term pair feature:
    [power_norm(|a−b|),  power_norm(a⊙b),  power_norm(‖a−b‖)]
    """
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    a, b = f[ia], f[ib]
    return np.concatenate([
        pnorm(np.abs(a - b)),
        pnorm(a * b),
        pnorm(np.linalg.norm(a - b, axis=1, keepdims=True)),
    ], axis=1).astype(np.float32)


# ─── LDZP similarity scores (3 scalars per pair, no SVM) ─────────────────────
def ldzp_similarity_scores(feats: np.ndarray,
                            ia: np.ndarray,
                            ib: np.ndarray) -> np.ndarray:
    """
    Compute 3 parameter-free similarity scores between LDZP histogram vectors.
    These are the standard metrics used in the LBP/LDP literature.

    Score 1 — Cosine similarity
        Measures global orientation match between the two histogram vectors.
        Range: [−1, 1], higher = more similar.

    Score 2 — Negative L2 distance (negated so higher = more similar)
        Measures overall magnitude difference.
        Range: (−∞, 0], closer to 0 = more similar.

    Score 3 — Negative Chi-squared distance
        THE standard metric for histogram comparison. For two histograms p, q:
          χ²(p,q) = Σ (p_i − q_i)² / (p_i + q_i + ε)
        Negated so higher = more similar.
        This is specifically designed for non-negative histogram features.

    Returns
    -------
    scores : float32 array of shape (N, 3)
        One row per pair, three similarity scores.
        No labels used — zero leakage risk.
    """
    a = feats[ia].astype(np.float64)
    b = feats[ib].astype(np.float64)

    # 1. Cosine similarity
    na     = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    nb     = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    cosine = ((a / na) * (b / nb)).sum(axis=1, keepdims=True)        # (N, 1)

    # 2. Negative L2 distance
    l2_neg = -np.linalg.norm(a - b, axis=1, keepdims=True)           # (N, 1)

    # 3. Negative Chi-squared distance
    #    χ²(p,q) = Σ (p_i − q_i)² / (p_i + q_i + ε)
    #    Power-normalize inputs first (standard for histogram chi2)
    ap = pnorm(np.abs(a), 0.5); bp = pnorm(np.abs(b), 0.5)
    chi2_neg = -(((ap - bp) ** 2) / (ap + bp + 1e-8)).sum(
        axis=1, keepdims=True)                                        # (N, 1)

    return np.concatenate([cosine, l2_neg, chi2_neg],
                          axis=1).astype(np.float32)                  # (N, 3)


def std_norm(Xtr: np.ndarray,
             Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler()
    return (sc.fit_transform(Xtr).astype(np.float32),
            sc.transform(Xte).astype(np.float32))


# ─── SE Block ─────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d, max(1, d // r)), nn.ReLU(),
            nn.Linear(max(1, d // r), d), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


# ─── Branch (one per modality) ────────────────────────────────────────────────
class Branch(nn.Module):
    def __init__(self, in_d, h=512, out=256, d1=0.15, d2=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, h),  nn.BatchNorm1d(h),   nn.ReLU(), nn.Dropout(d1),
            nn.Linear(h, out),   nn.BatchNorm1d(out),  nn.ReLU(),
            SEBlock(out),        nn.Dropout(d2))

    def forward(self, x):
        return self.net(x)


# ─── LCNN ─────────────────────────────────────────────────────────────────────
class LCNN(nn.Module):
    """
    5 branches: ArcFace / FaceNet / ResNet50 / VGGFace / LDZP-scores(3-d)
    Fusion head: 1280 → 768 → 512 (+residual) → 256 → 128 → 1

    Note: the LDZP branch receives only 3 numbers.
    Its Branch(3) maps: Linear(3,512)→BN→ReLU→DO→Linear(512,256)→BN→ReLU→SE→DO
    This is intentionally large relative to 3 inputs — it lets the network
    learn complex non-linear combinations of the 3 similarity scores.
    """
    def __init__(self, block_dims):
        super().__init__()
        self.bdims    = block_dims
        self.branches = nn.ModuleList([Branch(d) for d in block_dims])
        fd            = 256 * len(block_dims)                # 256 × 5 = 1280
        self.res      = nn.Sequential(nn.Linear(fd, 512), nn.BatchNorm1d(512))
        self.head     = nn.Sequential(
            nn.Linear(fd, 768),   nn.BatchNorm1d(768),  nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(768, 512),  nn.BatchNorm1d(512),  nn.ReLU(),           # idx 4,5,6
            nn.Linear(512, 256),  nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128),  nn.BatchNorm1d(128),  nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128, 1))

    def forward(self, x):
        sp    = torch.split(x, self.bdims, dim=1)
        fused = torch.cat([b(s) for b, s in zip(self.branches, sp)], dim=1)

        h = self.head[0](fused)   # Linear → 768
        h = self.head[1](h); h = self.head[2](h); h = self.head[3](h)
        h = self.head[4](h)       # Linear → 512
        h = self.head[5](h); h = self.head[6](h)
        h = h + self.res(fused)   # ← residual connection
        return self.head[7:](h).squeeze(1)


# ─── Training utilities ───────────────────────────────────────────────────────
def mixup(Xb, yb, a=0.3):
    lam = np.random.beta(a, a)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam * Xb + (1 - lam) * Xb[idx], lam * yb + (1 - lam) * yb[idx]


def train_ep(m, ld, opt, crit):
    m.train()
    for Xb, yb in ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        Xb, yb = mixup(Xb, yb)
        opt.zero_grad()
        loss = crit(m(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 2.0)
        opt.step()


def evl(m, ld):
    m.eval()
    p, l = [], []
    with torch.no_grad():
        for Xb, yb in ld:
            p.extend((torch.sigmoid(m(Xb.to(DEVICE))) > 0.5).cpu().numpy())
            l.extend(yb.numpy())
    return accuracy_score(l, p)


def train_lcnn(Xtr, ytr, Xva, yva, bdims,
               ep=300, bs=32, lr=3e-4, pat=35):
    tl = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(ytr, dtype=torch.float32)),
        batch_size=bs, shuffle=True)
    vl = DataLoader(
        TensorDataset(torch.tensor(Xva, dtype=torch.float32),
                      torch.tensor(yva, dtype=torch.float32)),
        batch_size=bs)

    m    = LCNN(bdims).to(DEVICE)
    pw   = torch.tensor(
        [(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
        dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = optim.AdamW(m.parameters(), lr=lr, weight_decay=2e-4)

    def lrf(e):
        return (e + 1) / 10 if e < 10 \
            else 0.5 * (1 + np.cos(np.pi * (e - 10) / max(1, ep - 10)))

    sch         = optim.lr_scheduler.LambdaLR(opt, lrf)
    best, state, w = 0, None, 0

    for e in range(ep):
        train_ep(m, tl, opt, crit)
        sch.step()
        acc = evl(m, vl)
        if acc > best:
            best  = acc
            state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            w     = 0
        else:
            w += 1
            if w >= pat:
                print(f"    Early stop epoch {e+1} | best val={best*100:.2f}%")
                break

    m.load_state_dict(state)
    return m


# ─── Main ─────────────────────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*65}")
    print(f"  {relation}")
    print(f"{'='*65}")

    arc  = load_emb(paths["arcface"])
    fn   = load_emb(paths["facenet"])
    rn   = load_emb(paths["resnet50"])
    vgg  = load_emb(paths["vggface"])
    ldzp = load_emb(paths["histldzp"])

    print(f"  ArcFace  : {arc.shape}   FaceNet : {fn.shape}")
    print(f"  ResNet50 : {rn.shape}  VGGFace : {vgg.shape}")
    print(f"  HistLDZP : {ldzp.shape}")

    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    # ── Pre-compute pair features (no labels used → zero leakage) ─────────
    arc_p  = deep_pair(arc,  idxa, idxb)   # (N, 1537)
    fn_p   = deep_pair(fn,   idxa, idxb)   # (N,  385)
    rn_p   = deep_pair(rn,   idxa, idxb)   # (N, 6145)
    vgg_p  = deep_pair(vgg,  idxa, idxb)   # (N, 1537)
    ldzp_s = ldzp_similarity_scores(ldzp, idxa, idxb)   # (N, 3)

    print(f"\n  Pair dims  arc:{arc_p.shape[1]}  fn:{fn_p.shape[1]}  "
          f"rn:{rn_p.shape[1]}  vgg:{vgg_p.shape[1]}  ldzp:{ldzp_s.shape[1]}")

    bdims      = [arc_p.shape[1], fn_p.shape[1],
                  rn_p.shape[1],  vgg_p.shape[1], ldzp_s.shape[1]]
    X_full     = np.concatenate([arc_p, fn_p, rn_p, vgg_p, ldzp_s], axis=1)

    fold_scores = []

    for f in range(1, 6):
        torch.manual_seed(42); np.random.seed(42); random.seed(42)

        tr, te   = fold != f, fold == f
        y_tr, y_te = y[tr], y[te]

        X_tr_sc, X_te_sc = std_norm(X_full[tr], X_full[te])

        fold_ids = fold[tr]
        vm       = fold_ids == f % 5 + 1

        m = train_lcnn(X_tr_sc[~vm], y_tr[~vm],
                       X_tr_sc[ vm], y_tr[ vm],
                       bdims)

        tl  = DataLoader(
            TensorDataset(torch.tensor(X_te_sc, dtype=torch.float32),
                          torch.tensor(y_te,    dtype=torch.float32)),
            batch_size=32)
        acc = evl(m, tl)
        fold_scores.append(acc)
        print(f"  Fold {f}: {acc*100:.2f}%")

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))
    all_results[relation] = {"mean_accuracy": mean_acc, "std_accuracy": std_acc}
    print(f"  ── Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  FINAL SUMMARY  —  Deep + HistLDZP scores → LCNN")
print(f"{'='*65}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for rel, res in all_results.items():
    print(f"{rel:<22} {res['mean_accuracy']*100:>9.2f}%"
          f"  ±{res['std_accuracy']*100:>6.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall : {overall*100:.2f}%")
print(f"  Target  : ≥ 90.00%")
print(f"  vs baseline (HistLBP) : {(overall - 0.88)*100:+.2f}%")