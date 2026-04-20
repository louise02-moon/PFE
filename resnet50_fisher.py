import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import random
import os

# ─── Reproducibility ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────
# Switch between the two models by changing MODE:
#   "resnet50"  → VGGFace2 ResNet50 (2048-dim)
#   "deepface"  → DeepFace VGG-Face (4096-dim)

MODE = "resnet50"   # change to "deepface" for the other model

if MODE == "resnet50":
    VGGFACE_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface2_resnet50_embeddings"
    FILE_PATTERN = "VGGFace2_{rel}_resnet50.pkl"
    TOP_K_GRID   = [50, 100, 200, 300, 500, 700, 1000, 1500, 2048]
else:
    VGGFACE_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\deepface_vggface_embeddings"
    FILE_PATTERN = "DeepFaceVGG_{rel}.pkl"
    TOP_K_GRID   = [100, 200, 300, 500, 700, 850, 1000, 1250, 1500]

MAT_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": "FD",
    "Father-Son"     : "FS",
    "Mother-Daughter": "MD",
    "Mother-Son"     : "MS",
}

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# ─── Fisher Score ─────────────────────────────────────────────────────────────
def fisher_score(X, y):
    scores = np.zeros(X.shape[1], dtype=np.float64)
    for c in np.unique(y):
        Xc   = X[y == c]
        Xnot = X[y != c]
        num  = (Xc.mean(axis=0) - Xnot.mean(axis=0)) ** 2
        den  = Xc.var(axis=0) + Xnot.var(axis=0) + 1e-8
        scores += num / den
    return scores

def select_top_k(X_train, X_test, y_train, k):
    idx = np.argsort(fisher_score(X_train, y_train))[::-1][:k]
    return X_train[:, idx], X_test[:, idx]

# ─── Power normalization ──────────────────────────────────────────────────────
def power_normalize(X_train, X_test):
    X_train = np.sqrt(np.abs(X_train))
    X_test  = np.sqrt(np.abs(X_test))
    mu      = X_train.mean(axis=0)
    return X_train - mu, X_test - mu

# ─── Inner CV ─────────────────────────────────────────────────────────────────
def find_best_params(X_raw, y, fold_ids, c_grid, k_grid):
    best_C, best_k, best_score = c_grid[0], k_grid[0], -1.0
    for k in k_grid:
        for C in c_grid:
            scores = []
            for inner_f in np.unique(fold_ids):
                Xtr = X_raw[fold_ids != inner_f]
                Xva = X_raw[fold_ids == inner_f]
                ytr = y[fold_ids != inner_f]
                yva = y[fold_ids == inner_f]

                Xtr_s, Xva_s = select_top_k(Xtr, Xva, ytr, k)
                Xtr_s, Xva_s = power_normalize(Xtr_s, Xva_s)

                clf = SVC(kernel="linear", C=C, random_state=42)
                clf.fit(Xtr_s, ytr)
                scores.append(accuracy_score(yva, clf.predict(Xva_s)))

            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score, best_C, best_k = mean_s, C, k
    return best_C, best_k

# ─── Main ─────────────────────────────────────────────────────────────────────
all_results = {}

print(f"\n  Running in MODE = '{MODE}'")

for relation, rel_code in RELATIONS.items():
    print(f"\n{'='*65}")
    print(f"  Relation : {relation}")
    print(f"{'='*65}")

    pkl_path = os.path.join(VGGFACE_DIR, FILE_PATTERN.format(rel=rel_code))

    import os
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    feats     = np.array(payload["features"], dtype=np.float64)
    filenames = payload["filenames"]

    print(f"  Layer         : {payload.get('layer', 'unknown')}")
    print(f"  Feature shape : {feats.shape}")
    print(f"  Feature range : [{feats.min():.4f}, {feats.max():.4f}]")
    print(f"  First file    : {filenames[0]}")
    print(f"  Last file     : {filenames[-1]}")

    # L2 normalize
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    mat  = sio.loadmat(f"{MAT_DIR}\\LBP_{rel_code.lower()}.mat")
    idxa = mat["idxa"].flatten() - 1
    idxb = mat["idxb"].flatten() - 1
    fold = mat["fold"].flatten()
    y    = mat["matches"].flatten()

    X = np.abs(feats[idxa] - feats[idxb])
    print(f"  Pair shape    : {X.shape}")

    fold_scores = []
    chosen_Cs   = []
    chosen_ks   = []

    for f in range(1, 6):
        train_mask  = fold != f
        test_mask   = fold == f

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y[train_mask]
        y_test      = y[test_mask]
        fold_ids    = fold[train_mask]

        best_C, best_k = find_best_params(
            X_train_raw, y_train, fold_ids, C_GRID, TOP_K_GRID
        )
        chosen_Cs.append(best_C)
        chosen_ks.append(best_k)

        X_train_s, X_test_s = select_top_k(X_train_raw, X_test_raw, y_train, best_k)
        X_train_s, X_test_s = power_normalize(X_train_s, X_test_s)

        clf = SVC(kernel="linear", C=best_C, random_state=42)
        clf.fit(X_train_s, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test_s))
        fold_scores.append(acc)
        print(f"  Fold {f}: {acc*100:.2f}%   (C={best_C}, k={best_k})")

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))

    all_results[relation] = {
        "fold_scores"   : fold_scores,
        "mean_accuracy" : mean_acc,
        "std_accuracy"  : std_acc,
        "best_Cs"       : chosen_Cs,
        "best_ks"       : chosen_ks,
    }
    print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FINAL SUMMARY — {MODE.upper()} + Fisher Score + Linear SVM")
print(f"{'='*65}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall ({MODE}) + Fisher + SVM : {overall*100:.2f}%")
print(f"  Reference — Hist-LBP alone      : 88.00%")
print(f"  Reference — paper target        : ~84.00%")