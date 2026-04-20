import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random

# ─── Reproducibility ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
ARCFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
FACENET_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\FaceNet\facenet_embeddings"
VGGFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface_embeddings"
HLBP_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FD.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_FD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FS.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_FS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MD.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_MD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MS.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_MS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_ms.mat",
    },
}

# ─── Per-relation configuration ───────────────────────────────────────────────
# FD is cross-gender — harder, benefits from more conservative settings
# FS, MD, MS are same-gender — stronger signal, can use richer features
RELATION_CONFIG = {
    "Father-Daughter": {
        "kernel"       : "rbf",
        "C_grid"       : [0.001, 0.01, 0.1, 1, 10, 100],
        "use_deep"     : True,
        "deep_weight"  : 0.5,    # reduce deep feature contribution for cross-gender
        "hlbp_weight"  : 2.0,    # boost Hist-LBP which is more robust for FD
        "fusion_modes" : ["diff", "prod", "sq_diff"],  # skip cosine for FD
    },
    "Father-Son": {
        "kernel"       : "rbf",
        "C_grid"       : [0.001, 0.01, 0.1, 1, 10, 100],
        "use_deep"     : True,
        "deep_weight"  : 1.0,
        "hlbp_weight"  : 1.0,
        "fusion_modes" : ["diff", "prod", "sq_diff", "cosine"],
    },
    "Mother-Daughter": {
        "kernel"       : "rbf",
        "C_grid"       : [0.001, 0.01, 0.1, 1, 10, 100],
        "use_deep"     : True,
        "deep_weight"  : 1.0,
        "hlbp_weight"  : 1.0,
        "fusion_modes" : ["diff", "prod", "sq_diff", "cosine"],
    },
    "Mother-Son": {
        "kernel"       : "rbf",
        "C_grid"       : [0.001, 0.01, 0.1, 1, 10, 100],
        "use_deep"     : True,
        "deep_weight"  : 0.7,
        "hlbp_weight"  : 1.5,    # slight boost for cross-gender
        "fusion_modes" : ["diff", "prod", "sq_diff", "cosine"],
    },
}

# ─── Load embedding ───────────────────────────────────────────────────────────
def load_embedding(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "features" not in data:
        sorted_keys = sorted(data.keys())
        feats = np.array([data[k] for k in sorted_keys], dtype=np.float64)
    elif isinstance(data, dict) and "features" in data:
        feats = np.array(data["features"], dtype=np.float64)
    else:
        feats = np.array(data, dtype=np.float64)
    return feats

# ─── Build pair features with configurable fusion modes ──────────────────────
def deep_pair_features(feats, idxa, idxb, fusion_modes, weight=1.0):
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    a, b  = feats[idxa], feats[idxb]

    parts = []
    if "diff"    in fusion_modes: parts.append(np.abs(a - b))
    if "prod"    in fusion_modes: parts.append(a * b)
    if "sq_diff" in fusion_modes: parts.append((a - b) ** 2)
    if "cosine"  in fusion_modes: parts.append(np.sum(a * b, axis=1, keepdims=True))

    result = np.concatenate(parts, axis=1)
    return result * weight

def hlbp_pair_features(feats, idxa, idxb, weight=1.0):
    feats = np.sqrt(np.abs(feats))
    a, b  = feats[idxa], feats[idxb]
    diff  = np.abs(a - b)
    diff  = diff - diff.mean(axis=0)
    return diff * weight

# ─── Normalization ────────────────────────────────────────────────────────────
def normalize(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)

# ─── Inner CV ─────────────────────────────────────────────────────────────────
def find_best_C(X_raw, y, fold_ids, c_grid, kernel):
    best_C, best_score = c_grid[0], -1.0
    for C in c_grid:
        scores = []
        for inner_f in np.unique(fold_ids):
            Xtr = X_raw[fold_ids != inner_f]
            Xva = X_raw[fold_ids == inner_f]
            ytr = y[fold_ids != inner_f]
            yva = y[fold_ids == inner_f]
            Xtr_n, Xva_n = normalize(Xtr, Xva)
            clf = SVC(kernel=kernel, C=C, gamma='scale', random_state=42)
            clf.fit(Xtr_n, ytr)
            scores.append(accuracy_score(yva, clf.predict(Xva_n)))
        mean_s = float(np.mean(scores))
        if mean_s > best_score:
            best_score, best_C = mean_s, C
    return best_C

# ─── Main ─────────────────────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    cfg = RELATION_CONFIG[relation]

    print(f"\n{'='*65}")
    print(f"  Relation : {relation}")
    print(f"  Config   : kernel={cfg['kernel']}, "
          f"deep_w={cfg['deep_weight']}, hlbp_w={cfg['hlbp_weight']}, "
          f"fusion={cfg['fusion_modes']}")
    print(f"{'='*65}")

    arcface_feats = load_embedding(paths["arcface"])
    facenet_feats = load_embedding(paths["facenet"])
    vggface_feats = load_embedding(paths["vggface"])
    hlbp_feats    = load_embedding(paths["hlbp"])

    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    # Build pair features using relation-specific config
    arc_X  = deep_pair_features(arcface_feats, idxa, idxb,
                                cfg["fusion_modes"], cfg["deep_weight"])
    fn_X   = deep_pair_features(facenet_feats, idxa, idxb,
                                cfg["fusion_modes"], cfg["deep_weight"])
    vgg_X  = deep_pair_features(vggface_feats, idxa, idxb,
                                cfg["fusion_modes"], cfg["deep_weight"])
    hlbp_X = hlbp_pair_features(hlbp_feats, idxa, idxb, cfg["hlbp_weight"])

    X = np.concatenate([arc_X, fn_X, vgg_X, hlbp_X], axis=1)
    print(f"  Combined shape : {X.shape}")

    fold_scores = []
    best_Cs     = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y[train_mask]
        y_test      = y[test_mask]

        best_C = find_best_C(X_train_raw, y_train,
                             fold[train_mask], cfg["C_grid"], cfg["kernel"])
        best_Cs.append(best_C)

        X_train, X_test = normalize(X_train_raw, X_test_raw)

        clf = SVC(kernel=cfg["kernel"], C=best_C,
                  gamma='scale', random_state=42)
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        fold_scores.append(acc)
        print(f"  Fold {f}: {acc*100:.2f}%   (C={best_C})")

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))

    all_results[relation] = {
        "fold_scores"   : fold_scores,
        "mean_accuracy" : mean_acc,
        "std_accuracy"  : std_acc,
        "best_Cs"       : best_Cs,
    }
    print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  FINAL SUMMARY — Per-Relation Tuned Pipeline")
print(f"{'='*65}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall Per-Relation Tuned : {overall*100:.2f}%")
print(f"  Reference — best so far    : 88.30%")
print(f"\n  Per-relation config used:")
for relation, cfg in RELATION_CONFIG.items():
    print(f"  {relation:<22} deep_w={cfg['deep_weight']}, "
          f"hlbp_w={cfg['hlbp_weight']}, "
          f"modes={len(cfg['fusion_modes'])}")