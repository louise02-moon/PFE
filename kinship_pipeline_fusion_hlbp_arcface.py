import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import random

# Reproductibilité
random.seed(42)
np.random.seed(42)

# ─── Chemins des fichiers ────────────────────────────────────────────────────
ARCFACE_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
HLBP_DIR    = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_FD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_FS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_MD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_MS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID = [0.01, 0.1, 1, 10, 100, 1000]

# ─── Power normalization ──────────────────────────────────────────────────────
def power_normalization(X_train, X_test):
    X_train = np.sqrt(np.abs(X_train))
    X_test  = np.sqrt(np.abs(X_test))
    mean_   = X_train.mean(axis=0)
    X_train = X_train - mean_
    X_test  = X_test  - mean_
    return X_train, X_test

# ─── Best C by inner CV ───────────────────────────────────────────────────────
def find_best_C(X_train_raw, y_train, train_fold_ids, c_grid):
    inner_folds = np.unique(train_fold_ids)
    best_C, best_score = c_grid[0], -1

    for C in c_grid:
        scores = []
        for inner_f in inner_folds:
            imask_tr = train_fold_ids != inner_f
            imask_te = train_fold_ids == inner_f

            Xtr, Xte = X_train_raw[imask_tr], X_train_raw[imask_te]
            ytr, yte = y_train[imask_tr],     y_train[imask_te]

            Xtr, Xte = power_normalization(Xtr, Xte)

            clf = SVC(kernel='linear', C=C, random_state=42)
            clf.fit(Xtr, ytr)
            scores.append(accuracy_score(yte, clf.predict(Xte)))

        mean_s = np.mean(scores)
        if mean_s > best_score:
            best_score, best_C = mean_s, C

    return best_C

# ─── Boucle principale ───────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*55}")
    print(f"  Relation : {relation}")
    print(f"{'='*55}")

    # ── Load ArcFace embeddings (dict: filename → embedding) ─────────────────
    with open(paths["arcface"], "rb") as f:
        arcface_dict = pickle.load(f)

    sorted_keys  = sorted(arcface_dict.keys())
    arcface_feats = np.array([arcface_dict[k] for k in sorted_keys],
                              dtype=np.float64)

    # L2 normalize ArcFace
    arcface_feats = arcface_feats / (
        np.linalg.norm(arcface_feats, axis=1, keepdims=True) + 1e-8
    )

    # ── Load Hist-LBP features (numpy array) ─────────────────────────────────
    with open(paths["hlbp"], "rb") as f:
        hlbp_feats = pickle.load(f)
    hlbp_feats = np.array(hlbp_feats, dtype=np.float64)

    print(f"  ArcFace shape : {arcface_feats.shape}")
    print(f"  Hist-LBP shape: {hlbp_feats.shape}")

    # Safety check — both must have same number of images
    assert arcface_feats.shape[0] == hlbp_feats.shape[0], \
        f"Mismatch! ArcFace has {arcface_feats.shape[0]} images " \
        f"but Hist-LBP has {hlbp_feats.shape[0]} images."

    # ── Concatenate ArcFace + Hist-LBP into one feature matrix ───────────────
    # ArcFace: 512 dims — deep identity features
    # Hist-LBP: 2832 dims — texture + color features
    # Combined: 3344 dims
    combined = np.concatenate([arcface_feats, hlbp_feats], axis=1)
    print(f"  Combined shape: {combined.shape}")

    # ── Load metadata ─────────────────────────────────────────────────────────
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    matches = mat['matches'].flatten()

    # ── Absolute difference on combined features ──────────────────────────────
    X = np.abs(combined[idxa] - combined[idxb])
    y = matches

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

        # Power normalization
        X_train, X_test = power_normalization(X_train_raw, X_test_raw)

        # Linear SVM
        clf = SVC(kernel='linear', C=best_C, random_state=42)
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        fold_scores.append(acc)
        print(f"  Fold {f}: {acc * 100:.2f}%   (C = {best_C})")

    mean_acc = np.mean(fold_scores)
    std_acc  = np.std(fold_scores)

    all_results[relation] = {
        "fold_scores"   : fold_scores,
        "mean_accuracy" : mean_acc,
        "std_accuracy"  : std_acc,
        "best_Cs"       : best_Cs,
    }
    print(f"  Mean : {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

# ─── Récapitulatif ────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  RÉCAPITULATIF FINAL")
print(f"{'='*55}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")

overall_acc = np.mean([res["mean_accuracy"] for res in all_results.values()])

print(f"\n{'='*55}")
print("  ACCURACY GLOBALE — ArcFace + Hist-LBP Fusion + Linear SVM")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  ArcFace + Hist-LBP Fusion Accuracy : {overall_acc * 100:.2f}%")