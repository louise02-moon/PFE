import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random

# Reproductibilité
random.seed(42)
np.random.seed(42)

# ─── Chemins des fichiers ────────────────────────────────────────────────────
PKL_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
MAT_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "pkl": f"{PKL_DIR}\\ArcFace_FD.pkl",
        "mat": f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "pkl": f"{PKL_DIR}\\ArcFace_FS.pkl",
        "mat": f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "pkl": f"{PKL_DIR}\\ArcFace_MD.pkl",
        "mat": f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "pkl": f"{PKL_DIR}\\ArcFace_MS.pkl",
        "mat": f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]

# ─── Pair feature builder ─────────────────────────────────────────────────────
def build_pair_features(ux, idxa, idxb):
    """
    Builds a rich 1025-dim feature vector for each pair:
      1. |a - b|      — how different they are per dimension  (512 dims)
      2. a * b        — what they share per dimension         (512 dims)
      3. dot(a, b)    — overall cosine similarity score       (  1 dim)
    """
    a = ux[idxa]
    b = ux[idxb]

    # L2 normalize — ArcFace embeddings should already be normalized
    # but we do it anyway to be safe
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

    abs_diff  = np.abs(a - b)                              # (N, 512)
    elem_prod = a * b                                      # (N, 512)
    cos_sim   = np.sum(a * b, axis=1, keepdims=True)       # (N, 1)

    return np.concatenate([abs_diff, elem_prod, cos_sim], axis=1)  # (N, 1025)

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

            sc  = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

            clf = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
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

    # Chargement embeddings
    with open(paths["pkl"], "rb") as f:
        emb_dict = pickle.load(f)

    sorted_keys = sorted(emb_dict.keys())
    ux = np.array([emb_dict[k] for k in sorted_keys], dtype=np.float64)

    # Chargement métadonnées
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    matches = mat['matches'].flatten()

    # Richer pair features
    X = build_pair_features(ux, idxa, idxb)   # (N, 1025)
    y = matches

    print(f"  Pair feature shape : {X.shape}")

    fold_scores = []
    best_Cs     = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y[train_mask]
        y_test      = y[test_mask]

        # StandardScaler — fit on train only
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        # Best C by inner CV
        best_C = find_best_C(X_train_raw, y_train, fold[train_mask], C_GRID)
        best_Cs.append(best_C)

        # Final SVM
        clf = SVC(kernel='rbf', C=best_C, gamma='scale', random_state=42)
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
print("  ACCURACY GLOBALE — ArcFace + SVM (cosine features)")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  ArcFace + SVM (cosine) Accuracy : {overall_acc * 100:.2f}%")