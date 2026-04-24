[24/04/2026 18:16] Karima: import os, pickle, random, time
import numpy as np
import scipy.io as sio
import cv2
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
print(f"Device : {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# ①  USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH   = r"C:\Users\Mazouni\Desktop\Karima\PFE\KinFaceW-II\images"
OUTPUT_DIR     = r"C:\Users\Mazouni\Desktop\Karima\PFE\Methodes classiques\HistZigZag\features"
MAT_DIR        = r"C:\Users\Mazouni\Desktop\Karima\PFE\lbp"

ARCFACE_DIR  = r"C:\Users\Mazouni\Desktop\Karima\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
FACENET_DIR  = r"C:\Users\Mazouni\Desktop\Karima\PFE\Apprentissage profond\FaceNet\facenet_embeddings"
RESNET50_DIR = r"C:\Users\Mazouni\Desktop\Karima\PFE\Apprentissage profond\ResNet50\resnet50_embeddings"
VGGFACE_DIR  = r"C:\Users\Mazouni\Desktop\Karima\PFE\Apprentissage profond\VGGFace\vggface_embeddings"

EXTRACT    = False

PATCH_SIZE = 16
STEP_SIZE  = 8
NUM_BINS   = 59
IMAGE_SIZE = (64, 64)

RELATIONS = ["FD", "FS", "MD", "MS"]
REL_NAMES = {"FD": "Father-Daughter", "FS": "Father-Son",
             "MD": "Mother-Daughter", "MS": "Mother-Son"}

DEEP_PATHS = {
    "FD": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FD.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_FD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGG16_FD.pkl",
    },
    "FS": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FS.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_FS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGG16_FS.pkl",
    },
    "MD": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MD.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_MD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGG16_MD.pkl",
    },
    "MS": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MS.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_MS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGG16_MS.pkl",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# ②  HYPERPARAMÈTRES SPÉCIALISÉS PAR RELATION
#     FD/FS  : relations père  → paramètres proches de l'original (85%)
#     MD/MS  : relations mère  → LR plus élevé, dropout réduit,
#                                mixup doux, patience plus longue
# ══════════════════════════════════════════════════════════════════════════════
[24/04/2026 18:16] Karima: REL_HP = {
    "FD": dict(
        lr           = 3e-4,
        weight_decay = 5e-4,
        epochs       = 600,
        patience     = 50,
        mixup_alpha  = 0.4,
        drop_deep    = 0.40,
        drop_zz      = 0.35,
        drop_cls1    = 0.35,
        drop_cls2    = 0.25,
        se_r         = 4,
    ),
    "FS": dict(
        lr           = 3e-4,
        weight_decay = 5e-4,
        epochs       = 600,
        patience     = 30,
        mixup_alpha  = 0.4,
        drop_deep    = 0.40,
        drop_zz      = 0.35,
        drop_cls1    = 0.35,
        drop_cls2    = 0.25,
        se_r         = 4,
    ),
    # ── MD : mère-fille, inter-genre plus difficile ──────────────────────
    "MD": dict(
        lr           = 5e-4,   # LR plus élevé pour converger sur MD
        weight_decay = 3e-4,   # moins de régularisation
        epochs       = 700,    # plus d'epochs
        patience     = 60,     # plus de patience
        mixup_alpha  = 0.2,    # mixup doux : paires MD déjà difficiles
        drop_deep    = 0.30,   # dropout réduit : évite sous-apprentissage
        drop_zz      = 0.25,
        drop_cls1    = 0.25,
        drop_cls2    = 0.15,
        se_r         = 4,
    ),
    # ── MS : mère-fils, relation la plus difficile ───────────────────────
    "MS": dict(
        lr           = 5e-4,   # même logique que MD
        weight_decay = 3e-4,
        epochs       = 700,
        patience     = 60,
        mixup_alpha  = 0.2,    # mixup très doux
        drop_deep    = 0.30,
        drop_zz      = 0.25,
        drop_cls1    = 0.25,
        drop_cls2    = 0.15,
        se_r         = 4,
    ),
}

BATCH = 32

# ══════════════════════════════════════════════════════════════════════════════
# ③  SPLIT  80 / 10 / 10
# ══════════════════════════════════════════════════════════════════════════════

def split_fold_half(fold_mask: np.ndarray, seed: int = 42):
    indices = np.where(fold_mask)[0]
    rng     = np.random.RandomState(seed)
    rng.shuffle(indices)
    mid     = len(indices) // 2
    mask_te = np.zeros(len(fold_mask), dtype=bool)
    mask_va = np.zeros(len(fold_mask), dtype=bool)
    mask_te[indices[:mid]]  = True
    mask_va[indices[mid:]]  = True
    return mask_te, mask_va

# ══════════════════════════════════════════════════════════════════════════════
# ④  PRÉCALCUL ZigZag + LBP uniforme
# ══════════════════════════════════════════════════════════════════════════════

def zigzag_indices(n: int) -> np.ndarray:
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n-1), max(0, s-n+1)-1, -1):
                indices.append((i, s - i))
        else:
            for i in range(max(0, s-n+1), min(s, n-1)+1):
                indices.append((i, s - i))
    return np.array(indices)

def build_uniform_table(num_bins: int) -> np.ndarray:
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

# ══════════════════════════════════════════════════════════════════════════════
# ⑤  LBP VECTORISÉ
# ══════════════════════════════════════════════════════════════════════════════

_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]

def lbp_fast(channel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    lbp  = np.zeros((h, w), dtype=np.uint8)
    c    = channel[1:-1, 1:-1]
    for bit, (dr, dc) in enumerate(_OFFSETS):
        neighbor = channel[1+dr:h-1+dr, 1+dc:w-1+dc]
        lbp[1:-1, 1:-1] |= np.uint8((neighbor >= c).astype(np.uint8) << bit)
    return lbp

# ══════════════════════════════════════════════════════════════════════════════
# ⑥  EXTRACTION PATCHES + HISTOGRAMMES ZIGZAG
# ══════════════════════════════════════════════════════════════════════════════
[24/04/2026 18:16] Karima: def extract_patches(lbp_map: np.ndarray) -> np.ndarray:
    h, w     = lbp_map.shape
    rows     = range(0, h - PATCH_SIZE + 1, STEP_SIZE)
    cols     = range(0, w - PATCH_SIZE + 1, STEP_SIZE)
    s_h, s_w = lbp_map.strides
    shape    = (len(rows), len(cols), PATCH_SIZE, PATCH_SIZE)
    strides  = (s_h * STEP_SIZE, s_w * STEP_SIZE, s_h, s_w)
    patches  = np.lib.stride_tricks.as_strided(lbp_map, shape=shape, strides=strides)
    return patches.reshape(len(rows) * len(cols), PATCH_SIZE, PATCH_SIZE)

def patches_to_hists(patches: np.ndarray) -> np.ndarray:
    r, c  = ZZ_IDX[:, 0], ZZ_IDX[:, 1]
    zz    = patches[:, r, c].astype(np.int32)
    N     = patches.shape[0]
    hists = np.zeros((N, NUM_BINS), dtype=np.float32)
    for i in range(N):
        np.add.at(hists[i], zz[i], 1)
    hists /= (hists.sum(axis=1, keepdims=True) + 1e-8)
    return hists

def extract_histzigzag(image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] {image_path}"); return None
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

# ══════════════════════════════════════════════════════════════════════════════
# ⑦  RUN EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def run_extraction() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[EXTRACT]  patch={PATCH_SIZE}  step={STEP_SIZE}  bins={NUM_BINS}")
    t0 = time.time()
    for rel in RELATIONS:
        rel_path = os.path.join(DATASET_PATH, rel)
        if not os.path.isdir(rel_path):
            print(f"  [ERREUR] dossier introuvable : {rel_path}"); continue
        files   = sorted([f for f in os.listdir(rel_path)
                          if f.lower().endswith((".jpg",".png",".jpeg"))])
        vectors = []; t1 = time.time()
        for k, fname in enumerate(files):
            vec = extract_histzigzag(os.path.join(rel_path, fname))
            if vec is not None: vectors.append(vec)
            if (k+1) % 50 == 0:
                eta = (time.time()-t1)/(k+1) * (len(files)-k-1)
                print(f"  {rel}  {k+1:>3}/{len(files)}  ETA {eta:.0f}s")
        if not vectors:
            print(f"  [WARN] aucun vecteur pour {rel}"); continue
        mat = np.array(vectors, dtype=np.float32)
        out = os.path.join(OUTPUT_DIR, f"HistZigZag_{rel}.pkl")
        with open(out, "wb") as f: pickle.dump(mat, f)
        print(f"  ✓ {rel}: {mat.shape}  → {out}")
    print(f"\nExtraction terminée en {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# ⑧  LOADERS DES FEATURES DEEP
# ══════════════════════════════════════════════════════════════════════════════

def load_deep_embedding(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "features" not in data:
        feats = np.array([data[k] for k in sorted(data.keys())], dtype=np.float64)
    elif isinstance(data, dict) and "features" in data:
        feats = np.array(data["features"], dtype=np.float64)
    else:
        feats = np.array(data, dtype=np.float64)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return (feats / (norms + 1e-8)).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# ⑨  CONSTRUCTION DES PAIRES
# ══════════════════════════════════════════════════════════════════════════════
[24/04/2026 18:16] Karima: def power_norm(X: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return np.sign(X) * (np.abs(X)  alpha)

def build_deep_pair(feats: np.ndarray, idxa, idxb) -> np.ndarray:
    a    = feats[idxa].astype(np.float64)
    b    = feats[idxb].astype(np.float64)
    diff = power_norm(np.abs(a - b))
    prod = power_norm(a * b)
    dist = power_norm(np.linalg.norm(a - b, axis=1, keepdims=True))
    return np.concatenate([diff, prod, dist], axis=1).astype(np.float32)

def build_zigzag_pair(feats: np.ndarray, idxa, idxb) -> np.ndarray:
    a    = feats[idxa].astype(np.float64)
    b    = feats[idxb].astype(np.float64)

    diff = np.sqrt(np.abs(a - b))
    diff = diff - diff.mean(axis=0)

    prod = a * b
    prod = np.sqrt(np.abs(prod)) * np.sign(prod)

    an   = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn   = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cos  = np.sum(an * bn, axis=1, keepdims=True)

    chi2 = (a - b)  2 / (np.abs(a) + np.abs(b) + 1e-8)
    chi2 = power_norm(chi2)

    return np.concatenate([diff, prod, cos, chi2], axis=1).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# ⑩  NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_splits(X_tr, X_va, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_va), sc.transform(X_te)

# ══════════════════════════════════════════════════════════════════════════════
# ⑪  ARCHITECTURE LCNN  — dropout configurable par relation
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def init(self, d, r=4):
        super().init()
        self.se = nn.Sequential(
            nn.Linear(d, max(1, d//r)), nn.ReLU(),
            nn.Linear(max(1, d//r), d), nn.Sigmoid())
    def forward(self, x): return x * self.se(x)


class DeepZigZagLCNN(nn.Module):
    def init(self, deep_total_dim: int, zigzag_dim: int,
                 out: int = 256,
                 drop_deep: float = 0.40,
                 drop_zz:   float = 0.35,
                 drop_cls1: float = 0.35,
                 drop_cls2: float = 0.25,
                 se_r:      int   = 4):
        super().init()
        self.deep_total_dim = deep_total_dim
        self.zigzag_dim     = zigzag_dim

        # ── Branche Deep ──────────────────────────────────────────────
        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_total_dim, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(drop_deep),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(drop_deep - 0.05),
            nn.Linear(512, out),
            nn.BatchNorm1d(out),  nn.ReLU(),
            SEBlock(out, r=se_r), nn.Dropout(drop_deep - 0.10),
        )
        self.deep_res = nn.Sequential(
            nn.Linear(deep_total_dim, out),
            nn.BatchNorm1d(out),
        )

        # ── Branche ZigZag ────────────────────────────────────────────
        self.zz_mlp = nn.Sequential(
            nn.Linear(zigzag_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(drop_zz),
            nn.Linear(512, out),
            nn.BatchNorm1d(out), nn.ReLU(),
            SEBlock(out, r=se_r), nn.Dropout(drop_zz - 0.05),
        )
        self.zz_res = nn.Sequential(
            nn.Linear(zigzag_dim, out),
            nn.BatchNorm1d(out),
        )

        # ── Classifieur final ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(out * 2, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop_cls1),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(drop_cls2),
            nn.Linear(64, 1),
        )
[24/04/2026 18:16] Karima: def forward(self, x):
        x_deep = x[:, :self.deep_total_dim]
        x_zz   = x[:, self.deep_total_dim:]
        h_deep = self.deep_mlp(x_deep) + self.deep_res(x_deep)
        h_zz   = self.zz_mlp(x_zz)    + self.zz_res(x_zz)
        return self.classifier(torch.cat([h_deep, h_zz], dim=1)).squeeze(1)

# ══════════════════════════════════════════════════════════════════════════════
# ⑫  MIXUP  — alpha configurable par relation
# ══════════════════════════════════════════════════════════════════════════════

def mixup(Xb, yb, alpha: float = 0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam*Xb + (1-lam)*Xb[idx], lam*yb + (1-lam)*yb[idx]

# ══════════════════════════════════════════════════════════════════════════════
# ⑬  BOUCLES TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, crit, mixup_alpha: float = 0.4):
    model.train()
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        Xb, yb = mixup(Xb, yb, alpha=mixup_alpha)
        opt.zero_grad()
        loss = crit(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

def evaluate(model, loader):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.extend(
                (torch.sigmoid(model(Xb.to(DEVICE))) > 0.5).cpu().numpy())
            labels.extend(yb.numpy())
    return accuracy_score(labels, preds)

def make_loader(X, y, shuffle=False):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=BATCH, shuffle=shuffle)

# ══════════════════════════════════════════════════════════════════════════════
# ⑭  ENTRAÎNEMENT LCNN — hyperparamètres par relation
# ══════════════════════════════════════════════════════════════════════════════

def train_lcnn(X_tr, y_tr, X_va, y_va,
               deep_total_dim: int, zigzag_dim: int,
               rel: str = "FD"):

    hp = REL_HP[rel]

    tl    = make_loader(X_tr, y_tr, shuffle=True)
    vl    = make_loader(X_va, y_va)

    model = DeepZigZagLCNN(
        deep_total_dim, zigzag_dim,
        drop_deep = hp["drop_deep"],
        drop_zz   = hp["drop_zz"],
        drop_cls1 = hp["drop_cls1"],
        drop_cls2 = hp["drop_cls2"],
        se_r      = hp["se_r"],
    ).to(DEVICE)

    pw    = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
        dtype=torch.float32).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt   = optim.AdamW(model.parameters(),
                        lr           = hp["lr"],
                        weight_decay = hp["weight_decay"],
                        betas        = (0.9, 0.999))

    epochs  = hp["epochs"]
    patience= hp["patience"]

    def lrf(ep):
        warmup = 10
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+np.cos(np.pi*(ep-warmup)/max(1, epochs-warmup)))

    sch              = optim.lr_scheduler.LambdaLR(opt, lrf)
    best_acc, best_state, wait = 0., None, 0

    for ep in range(epochs):
        train_epoch(model, tl, opt, crit, mixup_alpha=hp["mixup_alpha"])
        sch.step()
        acc = evaluate(model, vl)
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop ep.{ep+1} | best val={best_acc*100:.2f}%")
                break

    model.load_state_dict(best_state)
    return model

# ══════════════════════════════════════════════════════════════════════════════
# ⑮  EXPÉRIENCE  5-fold  (80 / 10 / 10)
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment() -> list | None:
    relation_accs = []
[24/04/2026 18:16] Karima: for rel in RELATIONS:
        hp = REL_HP[rel]
        print(f"\n{'='*65}")
        print(f"  Relation : {REL_NAMES[rel]}")
        print(f"  HP  lr={hp['lr']}  wd={hp['weight_decay']}  "
              f"mixup={hp['mixup_alpha']}  drop_deep={hp['drop_deep']}  "
              f"drop_zz={hp['drop_zz']}  patience={hp['patience']}")
        print(f"{'='*65}")

        # ── Chargement HistZigZag ────────────────────────────────────────
        pkl = os.path.join(OUTPUT_DIR, f"HistZigZag_{rel}.pkl")
        if not os.path.exists(pkl):
            print(f"  [WARN] manquant : {pkl}"); return None
        with open(pkl, "rb") as f:
            zz_feats = np.array(pickle.load(f), dtype=np.float32)

        # ── Chargement features deep ─────────────────────────────────────
        dp = DEEP_PATHS[rel]
        arc_feats = load_deep_embedding(dp["arcface"])
        fn_feats  = load_deep_embedding(dp["facenet"])
        rn_feats  = load_deep_embedding(dp["resnet50"])
        vgg_feats = load_deep_embedding(dp["vggface"])

        print(f"  HistZigZag : {zz_feats.shape}")
        print(f"  ArcFace    : {arc_feats.shape}")
        print(f"  FaceNet    : {fn_feats.shape}")
        print(f"  ResNet50   : {rn_feats.shape}")
        print(f"  VGGFace    : {vgg_feats.shape}")

        # ── Chargement .mat ──────────────────────────────────────────────
        mat_path = os.path.join(MAT_DIR, f"LBP_{rel.lower()}.mat")
        if not os.path.exists(mat_path):
            print(f"  [ERREUR] .mat introuvable : {mat_path}"); return None
        mat  = sio.loadmat(mat_path)
        idxa = mat['idxa'].flatten() - 1
        idxb = mat['idxb'].flatten() - 1
        fold = mat['fold'].flatten()
        y    = mat['matches'].flatten()

        # ── Construction des paires ──────────────────────────────────────
        arc_pairs = build_deep_pair(arc_feats, idxa, idxb)
        fn_pairs  = build_deep_pair(fn_feats,  idxa, idxb)
        rn_pairs  = build_deep_pair(rn_feats,  idxa, idxb)
        vgg_pairs = build_deep_pair(vgg_feats, idxa, idxb)
        zz_pairs  = build_zigzag_pair(zz_feats, idxa, idxb)

        X_deep = np.concatenate([arc_pairs, fn_pairs, rn_pairs, vgg_pairs], axis=1)
        X      = np.concatenate([X_deep, zz_pairs], axis=1)

        deep_total_dim = X_deep.shape[1]
        zigzag_dim     = zz_pairs.shape[1]

        print(f"\n  Paires deep (4 modèles) : {deep_total_dim} dims")
        print(f"  Paires HistZigZag       : {zigzag_dim} dims")
        print(f"  Vecteur X total         : {X.shape}")

        fold_scores = []

        for fi in range(1, 6):
            torch.manual_seed(42); np.random.seed(42); random.seed(42)

            fold_mask           = (fold == fi)
            test_mask, val_mask = split_fold_half(fold_mask, seed=42 + fi)
            train_mask          = ~fold_mask

            X_tr_raw = X[train_mask]; y_tr = y[train_mask]
            X_va_raw = X[val_mask];   y_va = y[val_mask]
            X_te_raw = X[test_mask];  y_te = y[test_mask]

            n = len(y)
            print(f"\n  Fold {fi}/5  →  "
                  f"train={len(y_tr)} ({100*len(y_tr)/n:.0f}%)  "
                  f"val={len(y_va)} ({100*len(y_va)/n:.0f}%)  "
                  f"test={len(y_te)} ({100*len(y_te)/n:.0f}%)")

            X_tr_sc, X_va_sc, X_te_sc = normalize_splits(
                X_tr_raw, X_va_raw, X_te_raw)

            model = train_lcnn(
                X_tr_sc, y_tr,
                X_va_sc, y_va,
                deep_total_dim = deep_total_dim,
                zigzag_dim     = zigzag_dim,
                rel            = rel,
            )

            acc = evaluate(model, make_loader(X_te_sc, y_te))
            fold_scores.append(acc)
            print(f"  → Fold {fi} Test Accuracy : {acc*100:.2f}%")
[24/04/2026 18:16] Karima: rel_acc = float(np.mean(fold_scores))
        rel_std = float(np.std(fold_scores))
        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  {REL_NAMES[rel]:<28}       ║")
        print(f"  ║  Accuracy : {rel_acc*100:.2f}% ± {rel_std*100:.2f}%            ║")
        print(f"  ╚══════════════════════════════════════════╝")
        relation_accs.append(rel_acc)

    return relation_accs

# ══════════════════════════════════════════════════════════════════════════════
# ⑯  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    for path, label in [(DATASET_PATH,"DATASET_PATH"), (MAT_DIR,"MAT_DIR")]:
        if not os.path.isdir(path):
            print(f"[ERREUR] {label} introuvable : {path}"); return

    if EXTRACT:
        run_extraction()

    print("\n[INFO] Vérification des fichiers features...")
    all_ok = True
    for rel in RELATIONS:
        pkl = os.path.join(OUTPUT_DIR, f"HistZigZag_{rel}.pkl")
        ok  = os.path.exists(pkl)
        print(f"  {rel}: {'✓' if ok else '✗ MANQUANT'}  {pkl}")
        if not ok: all_ok = False
    if not all_ok:
        print("\n[ERREUR] Mettez EXTRACT=True pour générer les features."); return

    print("\n" + "="*65)
    print("  HistZigZag + ArcFace + FaceNet + ResNet50 + VGGFace  v4")
    print("  Hyperparamètres spécialisés par relation :")
    for rel in RELATIONS:
        hp = REL_HP[rel]
        print(f"  {rel}  lr={hp['lr']}  mixup={hp['mixup_alpha']}"
              f"  drop_deep={hp['drop_deep']}  patience={hp['patience']}")
    print("  Classifieur : LCNN  |  Split 80 / 10 / 10")
    print("="*65)

    t0     = time.time()
    result = run_experiment()
    if result is None:
        print("SKIPPED — vérifiez les fichiers."); return

    overall = float(np.mean(result))

    print("\n" + "─"*65)
    print("  RÉSULTATS FINAUX")
    print("─"*65)
    for rel, acc in zip(RELATIONS, result):
        print(f"  {REL_NAMES[rel]:<22}  {acc*100:.2f}%")
    print("─"*65)
    print(f"  Overall Mean               {overall*100:.2f}%")
    print(f"  Baseline HistZigZag seul   81.00%")
    print(f"  Version v3                 85.00%")
    print(f"  Objectif                   90.00%")
    print(f"  Temps total                {(time.time()-t0)/60:.1f} min")
    print("─"*65)
    gain  = (overall - 0.85) * 100
    delta = (overall - 0.90) * 100
    print(f"  Gain vs v3 (85%)  : {gain:+.2f}%")
    marker = "✅" if overall >= 0.90 else "⚠️ "
    print(f"  {marker}  Écart vs objectif 90% : {delta:+.2f}%")
    print("─"*65)


if name == "main":
    main()