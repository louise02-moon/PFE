import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random
import os

random.seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
ARCFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
FACENET_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\FaceNet\facenet_embeddings"
RESNET50_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ResNet50\resnet50_embeddings"
HLPQ_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LPQ\Color_HLPQ_feature_vectors"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FD.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_FD.pkl",
        "hlpq"    : f"{HLPQ_DIR}\\HistLPQ_FD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FS.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_FS.pkl",
        "hlpq"    : f"{HLPQ_DIR}\\HistLPQ_FS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MD.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_MD.pkl",
        "hlpq"    : f"{HLPQ_DIR}\\HistLPQ_MD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MS.pkl",
        "resnet50": f"{RESNET50_DIR}\\ResNet50_MS.pkl",
        "hlpq"    : f"{HLPQ_DIR}\\HistLPQ_MS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]

# ─── Loaders ──────────────────────────────────────────────────────────────────
def load_deep(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "features" not in data:
        sorted_keys = sorted(data.keys(), key=lambda k: os.path.basename(k))
        feats = np.array([data[k] for k in sorted_keys], dtype=np.float64)
    elif isinstance(data, dict) and "features" in data:
        feats = np.array(data["features"], dtype=np.float64)
    else:
        feats = np.array(data, dtype=np.float64)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / (norms + 1e-8)

def load_classical(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data, dtype=np.float64)

# ─── Pair features ────────────────────────────────────────────────────────────
def power_normalize(X, alpha=0.5):
    return np.sign(X) * (np.abs(X) ** alpha)

def deep_pairs(feats, idxa, idxb):
    a, b = feats[idxa], feats[idxb]
    return np.concatenate([
        power_normalize(np.abs(a - b)),
        power_normalize(a * b),
        power_normalize(np.linalg.norm(a-b, axis=1, keepdims=True))
    ], axis=1)

def classical_pairs(feats, idxa, idxb):
    """Power norm + absolute difference — same as proven HistLBP approach"""
    a, b = feats[idxa], feats[idxb]
    diff = np.sqrt(np.abs(a - b))
    return diff - diff.mean(axis=0)

# ─── Normalization + Inner CV ─────────────────────────────────────────────────
def normalize(Xtr, Xte):
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)

def find_best_C(X_raw, y, fold_ids, c_grid):
    best_C, best_score = c_grid[0], -1.0
    for C in c_grid:
        scores = []
        for inner_f in np.unique(fold_ids):
            Xtr, Xva = X_raw[fold_ids != inner_f], X_raw[fold_ids == inner_f]
            ytr, yva = y[fold_ids != inner_f],     y[fold_ids == inner_f]
            Xtr, Xva = normalize(Xtr, Xva)
            clf = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
            clf.fit(Xtr, ytr)
            scores.append(accuracy_score(yva, clf.predict(Xva)))
        mean_s = float(np.mean(scores))
        if mean_s > best_score:
            best_score, best_C = mean_s, C
    return best_C

# ─── Main ─────────────────────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*65}")
    print(f"  Relation : {relation}")
    print(f"{'='*65}")

    arc  = load_deep(paths["arcface"])
    fn   = load_deep(paths["facenet"])
    rn   = load_deep(paths["resnet50"])
    hlpq = load_classical(paths["hlpq"])

    print(f"  ArcFace  shape : {arc.shape}")
    print(f"  FaceNet  shape : {fn.shape}")
    print(f"  ResNet50 shape : {rn.shape}")
    print(f"  HistLPQ  shape : {hlpq.shape}")

    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    X = np.concatenate([
        deep_pairs(arc,  idxa, idxb),
        deep_pairs(fn,   idxa, idxb),
        deep_pairs(rn,   idxa, idxb),
        classical_pairs(hlpq, idxa, idxb),
    ], axis=1)

    print(f"  Combined shape : {X.shape}")

    fold_scores = []
    best_Cs     = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_tr_raw = X[train_mask]
        X_te_raw = X[test_mask]
        y_train  = y[train_mask]
        y_test   = y[test_mask]

        best_C = find_best_C(X_tr_raw, y_train, fold[train_mask], C_GRID)
        best_Cs.append(best_C)

        X_tr, X_te = normalize(X_tr_raw, X_te_raw)

        clf = SVC(kernel='rbf', C=best_C, gamma='scale', random_state=42)
        clf.fit(X_tr, y_train)

        acc = accuracy_score(y_test, clf.predict(X_te))
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
print("  FINAL SUMMARY")
print("  ArcFace + FaceNet + ResNet50 + Hist-LPQ")
print("  Deep: Diff + Prod + Euclidean + Power Norm")
print("  Hist-LPQ: Diff + Power Norm")
print(f"{'='*65}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall : {overall*100:.2f}%")
print(f"  Reference — best so far (HistLBP) : 88.30%")