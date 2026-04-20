import os
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
RESNET50_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ResNet50\vggface2_resnet50_embeddings"
VGGFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface_embeddings"
HISTLBP_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FD.pkl",
        "resnet50": f"{RESNET50_DIR}\\VGGFace2_FD_resnet50.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FD.pkl",
        "histlbp" : f"{HISTLBP_DIR}\\HistLBP_FD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FS.pkl",
        "resnet50": f"{RESNET50_DIR}\\VGGFace2_FS_resnet50.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FS.pkl",
        "histlbp" : f"{HISTLBP_DIR}\\HistLBP_FS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MD.pkl",
        "resnet50": f"{RESNET50_DIR}\\VGGFace2_MD_resnet50.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MD.pkl",
        "histlbp" : f"{HISTLBP_DIR}\\HistLBP_MD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MS.pkl",
        "resnet50": f"{RESNET50_DIR}\\VGGFace2_MS_resnet50.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MS.pkl",
        "histlbp" : f"{HISTLBP_DIR}\\HistLBP_MS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]

# ─── Load deep embedding ──────────────────────────────────────────────────────
def load_deep_embedding(path):
    """
    Handles three formats:
      1. {filename: embedding}           — ArcFace, FaceNet, ResNet50
      2. {full_path: embedding}          — VGGFace (keys are full paths)
      3. {"features": array, ...}        — VGGFace2 style
      4. plain numpy array               — Hist-LBP style (not used here)
    Always sorts by basename so ordering matches .mat indices.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "features" not in data:
        # Sort by basename — handles both filename keys and full path keys
        sorted_keys = sorted(data.keys(), key=lambda k: os.path.basename(k))
        feats = np.array([data[k] for k in sorted_keys], dtype=np.float64)

    elif isinstance(data, dict) and "features" in data:
        feats = np.array(data["features"], dtype=np.float64)

    else:
        feats = np.array(data, dtype=np.float64)

    # L2 normalize
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / (norms + 1e-8)
    return feats

# ─── Load Hist-LBP (no L2 norm) ──────────────────────────────────────────────
def load_histlbp(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data, dtype=np.float64)

# ─── Power normalization ──────────────────────────────────────────────────────
def power_normalize(X, alpha=0.5):
    return np.sign(X) * (np.abs(X) ** alpha)

# ─── Pair features for deep embeddings ───────────────────────────────────────
def build_deep_pair_features(feats, idxa, idxb, alpha=0.5):
    a = feats[idxa]
    b = feats[idxb]
    diff = power_normalize(np.abs(a - b), alpha)
    prod = power_normalize(a * b,         alpha)
    dist = power_normalize(np.linalg.norm(a - b, axis=1, keepdims=True), alpha)
    return np.concatenate([diff, prod, dist], axis=1)

# ─── Pair features for Hist-LBP ──────────────────────────────────────────────
def build_hlbp_pair_features(feats, idxa, idxb):
    a    = feats[idxa]
    b    = feats[idxb]
    diff = np.abs(a - b)
    diff = np.sqrt(diff)
    diff = diff - diff.mean(axis=0)
    return diff

# ─── Normalization ────────────────────────────────────────────────────────────
def normalize(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)

# ─── Inner CV ─────────────────────────────────────────────────────────────────
def find_best_C(X_train_raw, y_train, fold_ids, c_grid):
    best_C, best_score = c_grid[0], -1.0
    for C in c_grid:
        scores = []
        for inner_f in np.unique(fold_ids):
            Xtr = X_train_raw[fold_ids != inner_f]
            Xva = X_train_raw[fold_ids == inner_f]
            ytr = y_train[fold_ids != inner_f]
            yva = y_train[fold_ids == inner_f]
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
    print(f"\n{'='*70}")
    print(f"  Relation : {relation}")
    print(f"{'='*70}")

    arcface_feats  = load_deep_embedding(paths["arcface"])
    facenet_feats  = load_deep_embedding(paths["facenet"])
    resnet50_feats = load_deep_embedding(paths["resnet50"])
    vggface_feats  = load_deep_embedding(paths["vggface"])
    histlbp_feats  = load_histlbp(paths["histlbp"])

    print(f"  ArcFace  shape : {arcface_feats.shape}")
    print(f"  FaceNet  shape : {facenet_feats.shape}")
    print(f"  ResNet50 shape : {resnet50_feats.shape}")
    print(f"  VGGFace  shape : {vggface_feats.shape}")
    print(f"  HistLBP  shape : {histlbp_feats.shape}")

    n = arcface_feats.shape[0]
    assert facenet_feats.shape[0]  == n, "FaceNet count mismatch!"
    assert resnet50_feats.shape[0] == n, "ResNet50 count mismatch!"
    assert vggface_feats.shape[0]  == n, "VGGFace count mismatch!"
    assert histlbp_feats.shape[0]  == n, "HistLBP count mismatch!"

    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    arc_pairs  = build_deep_pair_features(arcface_feats,  idxa, idxb)
    fn_pairs   = build_deep_pair_features(facenet_feats,  idxa, idxb)
    rn_pairs   = build_deep_pair_features(resnet50_feats, idxa, idxb)
    vgg_pairs  = build_deep_pair_features(vggface_feats,  idxa, idxb)
    lbp_pairs  = build_hlbp_pair_features(histlbp_feats,  idxa, idxb)

    X = np.concatenate([arc_pairs, fn_pairs, rn_pairs, vgg_pairs, lbp_pairs], axis=1)
    print(f"  Combined shape : {X.shape}")

    fold_scores = []
    best_Cs     = []

    for f in range(1, 6):
        train_mask  = fold != f
        test_mask   = fold == f

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y[train_mask]
        y_test      = y[test_mask]

        best_C = find_best_C(X_train_raw, y_train, fold[train_mask], C_GRID)
        best_Cs.append(best_C)

        X_train, X_test = normalize(X_train_raw, X_test_raw)

        clf = SVC(kernel='rbf', C=best_C, gamma='scale', random_state=42)
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
print(f"\n{'='*70}")
print("  FINAL SUMMARY")
print("  ArcFace + FaceNet + ResNet50 + VGGFace + Hist-LBP")
print("  Deep: Diff + Prod + Euclidean + Power Norm")
print("  Hist-LBP: Diff + Power Norm (proven pipeline)")
print(f"{'='*70}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall Accuracy : {overall*100:.2f}%")
print(f"\n  Reference — Hist-LBP alone                   : 88.00%")
print(f"  Reference — ArcFace+FaceNet+ResNet50+HistLBP : 88.30%")