import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import random

random.seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
HLPQ_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LPQ\HLPQ_feature_vectors"
MAT_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "hlpq": f"{HLPQ_DIR}\\HistLPQ_FD.pkl",
        "mat" : f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "hlpq": f"{HLPQ_DIR}\\HistLPQ_FS.pkl",
        "mat" : f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "hlpq": f"{HLPQ_DIR}\\HistLPQ_MD.pkl",
        "mat" : f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "hlpq": f"{HLPQ_DIR}\\HistLPQ_MS.pkl",
        "mat" : f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID = [0.01, 0.1, 1, 10, 100, 1000]

# ─── Power normalization ──────────────────────────────────────────────────────
def power_normalization(X_train, X_test):
    X_train = np.sqrt(np.abs(X_train))
    X_test  = np.sqrt(np.abs(X_test))
    mean_   = X_train.mean(axis=0)
    return X_train - mean_, X_test - mean_

# ─── Inner CV ─────────────────────────────────────────────────────────────────
def find_best_C(X_raw, y, fold_ids, c_grid):
    best_C, best_score = c_grid[0], -1.0
    for C in c_grid:
        scores = []
        for inner_f in np.unique(fold_ids):
            Xtr, Xva = X_raw[fold_ids != inner_f], X_raw[fold_ids == inner_f]
            ytr, yva = y[fold_ids != inner_f],     y[fold_ids == inner_f]
            Xtr, Xva = power_normalization(Xtr, Xva)
            clf = SVC(kernel='linear', C=C, random_state=42)
            clf.fit(Xtr, ytr)
            scores.append(accuracy_score(yva, clf.predict(Xva)))
        mean_s = float(np.mean(scores))
        if mean_s > best_score:
            best_score, best_C = mean_s, C
    return best_C

# ─── Main ─────────────────────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*55}")
    print(f"  Relation : {relation}")
    print(f"{'='*55}")

    with open(paths["hlpq"], "rb") as f:
        feats = np.array(pickle.load(f), dtype=np.float64)

    print(f"  Feature shape : {feats.shape}")
    print(f"  Value range   : [{feats.min():.4f}, {feats.max():.4f}]")

    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    X = np.abs(feats[idxa] - feats[idxb])

    fold_scores = []
    best_Cs     = []

    for f in range(1, 6):
        train_mask  = fold != f
        test_mask   = fold == f

        X_tr_raw = X[train_mask]
        X_te_raw = X[test_mask]
        y_train  = y[train_mask]
        y_test   = y[test_mask]

        best_C = find_best_C(X_tr_raw, y_train, fold[train_mask], C_GRID)
        best_Cs.append(best_C)

        X_tr, X_te = power_normalization(X_tr_raw, X_te_raw)

        clf = SVC(kernel='linear', C=best_C, random_state=42)
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
print(f"\n{'='*55}")
print("  FINAL SUMMARY — Hist-LPQ + Linear SVM")
print(f"{'='*55}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall Hist-LPQ : {overall*100:.2f}%")
print(f"  Reference — Hist-LBP standalone : 88.00%")