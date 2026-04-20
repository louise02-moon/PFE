import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random
import os

# ─── Reproducibility ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
ARCFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
FACENET_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\FaceNet\facenet_embeddings"
VGGFACE_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface_embeddings"
HLBP_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "arcface"  : f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "facenet"  : f"{FACENET_DIR}\\FaceNet_FD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FD.pkl",
        "hlbp"     : f"{HLBP_DIR}\\HistLBP_FD.pkl",
        "mat"      : f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "arcface"  : f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "facenet"  : f"{FACENET_DIR}\\FaceNet_FS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FS.pkl",
        "hlbp"     : f"{HLBP_DIR}\\HistLBP_FS.pkl",
        "mat"      : f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "arcface"  : f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "facenet"  : f"{FACENET_DIR}\\FaceNet_MD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MD.pkl",
        "hlbp"     : f"{HLBP_DIR}\\HistLBP_MD.pkl",
        "mat"      : f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "arcface"  : f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "facenet"  : f"{FACENET_DIR}\\FaceNet_MS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MS.pkl",
        "hlbp"     : f"{HLBP_DIR}\\HistLBP_MS.pkl",
        "mat"      : f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]

# ─── Load embedding — handles dict and array formats ─────────────────────────
def load_embedding(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Format 1: {filename: embedding} — ArcFace, FaceNet
    if isinstance(data, dict) and "features" not in data:
        sorted_keys = sorted(data.keys())
        feats = np.array([data[k] for k in sorted_keys], dtype=np.float64)

    # Format 2: {"features": array, ...} — VGGFace2
    elif isinstance(data, dict) and "features" in data:
        feats = np.array(data["features"], dtype=np.float64)

    # Format 3: plain numpy array — Hist-LBP
    else:
        feats = np.array(data, dtype=np.float64)

    return feats

# ─── Build pair features for deep embeddings ─────────────────────────────────
def deep_pair_features(feats, idxa, idxb):
    """
    4 fusion strategies for deep embeddings:
      1. |a - b|       — absolute difference
      2. a * b         — element-wise product
      3. (a - b)^2     — squared difference (euclidean-like)
      4. dot(a, b)     — cosine similarity score (1 dim)
    """
    # L2 normalize first
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    a = feats[idxa]
    b = feats[idxb]

    diff    = np.abs(a - b)
    prod    = a * b
    sq_diff = (a - b) ** 2
    cos_sim = np.sum(a * b, axis=1, keepdims=True)   # (N, 1)

    return np.concatenate([diff, prod, sq_diff, cos_sim], axis=1)

# ─── Build pair features for Hist-LBP ────────────────────────────────────────
def hlbp_pair_features(feats, idxa, idxb):
    # Power normalization
    feats = np.sqrt(np.abs(feats))

    a = feats[idxa]
    b = feats[idxb]

    return np.abs(a - b)

# ─── Normalization ────────────────────────────────────────────────────────────
def normalize(X_train, X_test):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test

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
    print(f"\n{'='*65}")
    print(f"  Relation : {relation}")
    print(f"{'='*65}")

    # Load all features
    arcface_feats  = load_embedding(paths["arcface"])
    facenet_feats  = load_embedding(paths["facenet"])
    vggface2_feats = load_embedding(paths["vggface"])
    hlbp_feats     = load_embedding(paths["hlbp"])

    print(f"  ArcFace   shape : {arcface_feats.shape}")
    print(f"  FaceNet   shape : {facenet_feats.shape}")
    print(f"  VGGFace2  shape : {vggface2_feats.shape}")
    print(f"  Hist-LBP  shape : {hlbp_feats.shape}")

    n = arcface_feats.shape[0]
    assert facenet_feats.shape[0]  == n, "FaceNet count mismatch!"
    assert vggface2_feats.shape[0] == n, "VGGFace count mismatch!"
    assert hlbp_feats.shape[0]     == n, "Hist-LBP count mismatch!"

    # Load metadata
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    y       = mat['matches'].flatten()

    # Build pair features per modality
    arc_pairs  = deep_pair_features(arcface_feats,  idxa, idxb)
    fn_pairs   = deep_pair_features(facenet_feats,  idxa, idxb)
    vgg_pairs  = deep_pair_features(vggface2_feats, idxa, idxb)
    hlbp_pairs = hlbp_pair_features(hlbp_feats,     idxa, idxb)

    # Mean center Hist-LBP on full dataset before fold split
    # (harmless since it's just centering, no label info used)
    hlbp_mean  = hlbp_pairs.mean(axis=0)
    hlbp_pairs = hlbp_pairs - hlbp_mean

    # Concatenate all modalities
    X = np.concatenate([arc_pairs, fn_pairs, vgg_pairs, hlbp_pairs], axis=1)
    print(f"  Combined shape  : {X.shape}")

    fold_scores = []
    best_Cs     = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y[train_mask]
        y_test      = y[test_mask]

        # Best C by inner CV
        best_C = find_best_C(X_train_raw, y_train, fold[train_mask], C_GRID)
        best_Cs.append(best_C)

        # Normalize + SVM
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
print(f"\n{'='*65}")
print("  FINAL SUMMARY")
print("  ArcFace + FaceNet + VGGFace + Hist-LBP — Multi-Fusion")
print(f"{'='*65}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall : {overall*100:.2f}%")
print(f"\n  Reference — Hist-LBP alone                    : 88.00%")
print(f"  Reference — Deep+Hist-LBP (ResNet50) best     : 88.30%")