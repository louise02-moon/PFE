import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import random

# Reproductibilité
random.seed(42)
np.random.seed(42)

# ─── Chemins ──────────────────────────────────────────────────────────────────
PKL_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "pkl": f"{PKL_DIR}\\HistLBP_FD.pkl",
        "mat": f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "pkl": f"{PKL_DIR}\\HistLBP_FS.pkl",
        "mat": f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "pkl": f"{PKL_DIR}\\HistLBP_MD.pkl",
        "mat": f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "pkl": f"{PKL_DIR}\\HistLBP_MS.pkl",
        "mat": f"{MAT_DIR}\\LBP_ms.mat",
    },
}


# ─── Power normalization ──────────────────────────────────────────────────────
def power_normalization(X_train, X_test):
    # Step 1 & 2: abs + sqrt
    X_train = np.sqrt(np.abs(X_train))
    X_test  = np.sqrt(np.abs(X_test))
    # Step 3: mean centering — fitted on train only
    mean_   = X_train.mean(axis=0)
    X_train = X_train - mean_
    X_test  = X_test  - mean_
    return X_train, X_test


# ─── Main loop ────────────────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*55}")
    print(f"  Relation : {relation}")
    print(f"{'='*55}")

    # Load features
    with open(paths["pkl"], "rb") as f:
        ux = pickle.load(f)
    ux = np.array(ux, dtype=np.float64)
    print(f"  Feature shape : {ux.shape}")

    # Load metadata
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat["idxa"].flatten() - 1
    idxb    = mat["idxb"].flatten() - 1
    fold    = mat["fold"].flatten()
    matches = mat["matches"].flatten()

    # Absolute difference pair features
    X = np.abs(ux[idxa] - ux[idxb])
    y = matches

    fold_scores = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_train = X[train_mask]
        X_test  = X[test_mask]
        y_train = y[train_mask]
        y_test  = y[test_mask]

        # Power normalization (fitted on train only)
        X_train, X_test = power_normalization(X_train, X_test)

        # LinearSVC uses the same liblinear solver as MATLAB's fitcsvm,
        # giving results closer to the MATLAB pipeline than SVC does.
        clf = LinearSVC(C=0.001, random_state=42, max_iter=5000)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        fold_scores.append(acc)
        print(f"  Fold {f}: {acc*100:.2f}%")

    mean_acc = np.mean(fold_scores)
    std_acc  = np.std(fold_scores)

    all_results[relation] = {
        "fold_scores":   fold_scores,
        "mean_accuracy": mean_acc,
        "std_accuracy":  std_acc,
    }
    print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# ─── Summary ──────────────────────────────────────────────────────────────────
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
print("  ACCURACY GLOBALE — Hist-LBP + Power Norm + LinearSVC")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  {'─'*50}")
print(f"  Global Accuracy : {overall_acc*100:.2f}%")