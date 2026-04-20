import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import random

# ─── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── File paths ───────────────────────────────────────────────────────────────
VGGFACE_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface_embeddings"
MAT_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_FD.pkl",
        "mat":     f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_FS.pkl",
        "mat":     f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_MD.pkl",
        "mat":     f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_MS.pkl",
        "mat":     f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID = [0.01, 0.1, 1, 10, 100, 1000]


# ─── Power normalization (fitted on train only) ───────────────────────────────
def power_normalize(X_train, X_test):
    X_train = np.sqrt(np.abs(X_train))
    X_test  = np.sqrt(np.abs(X_test))
    mean_   = X_train.mean(axis=0)
    X_train = X_train - mean_
    X_test  = X_test  - mean_
    return X_train, X_test


# ─── Best C via inner cross-validation ───────────────────────────────────────
def find_best_C(X_train_raw, y_train, train_fold_ids, c_grid):
    inner_folds = np.unique(train_fold_ids)
    best_C, best_score = c_grid[0], -1

    for C in c_grid:
        scores = []
        for inner_f in inner_folds:
            mask_tr = train_fold_ids != inner_f
            mask_te = train_fold_ids == inner_f

            Xtr, Xte = X_train_raw[mask_tr], X_train_raw[mask_te]
            ytr, yte = y_train[mask_tr],     y_train[mask_te]

            Xtr, Xte = power_normalize(Xtr, Xte)

            clf = SVC(kernel='linear', C=C, random_state=42)
            clf.fit(Xtr, ytr)
            scores.append(accuracy_score(yte, clf.predict(Xte)))

        mean_s = np.mean(scores)
        if mean_s > best_score:
            best_score, best_C = mean_s, C

    return best_C


# ─── Main loop ────────────────────────────────────────────────────────────────
def run_experiment():
    all_results = {}

    for relation, paths in RELATIONS.items():
        print(f"\n{'='*55}")
        print(f"  Relation : {relation}")
        print(f"{'='*55}")

        # Load VGG-Face embeddings
        with open(paths["vggface"], "rb") as f:
            vgg_dict = pickle.load(f)

        sorted_keys = sorted(vgg_dict.keys())
        vgg_feats   = np.array([vgg_dict[k] for k in sorted_keys], dtype=np.float64)

        # L2-normalize
        norms     = np.linalg.norm(vgg_feats, axis=1, keepdims=True)
        vgg_feats = vgg_feats / (norms + 1e-8)

        print(f"  VGG-Face shape : {vgg_feats.shape}")

        # Load metadata
        mat     = sio.loadmat(paths["mat"])
        idxa    = mat['idxa'].flatten() - 1
        idxb    = mat['idxb'].flatten() - 1
        fold    = mat['fold'].flatten()
        matches = mat['matches'].flatten()

        # Sanity check
        assert vgg_feats.shape[0] == (idxa.max() + 1) or \
               vgg_feats.shape[0] >= max(idxa.max(), idxb.max()) + 1, \
            f"Index mismatch: {vgg_feats.shape[0]} embeddings but mat needs index up to {max(idxa.max(), idxb.max())}"

        # Absolute difference feature vectors
        X = np.abs(vgg_feats[idxa] - vgg_feats[idxb])
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

            best_C = find_best_C(X_train_raw, y_train, fold[train_mask], C_GRID)
            best_Cs.append(best_C)

            X_train, X_test = power_normalize(X_train_raw, X_test_raw)

            clf = SVC(kernel='linear', C=best_C, random_state=42)
            clf.fit(X_train, y_train)

            acc = accuracy_score(y_test, clf.predict(X_test))
            fold_scores.append(acc)
            print(f"  Fold {f}: {acc * 100:.2f}%   (C = {best_C})")

        mean_acc = np.mean(fold_scores)
        std_acc  = np.std(fold_scores)

        all_results[relation] = {
            "fold_scores":   fold_scores,
            "mean_accuracy": mean_acc,
            "std_accuracy":  std_acc,
            "best_Cs":       best_Cs,
        }
        print(f"  Mean : {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  FINAL SUMMARY — VGG-Face + Linear SVM")
    print(f"{'='*55}")
    print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
    print("-" * 44)
    for relation, res in all_results.items():
        print(
            f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
            f" {res['std_accuracy']*100:>7.2f}%"
        )

    overall_acc = np.mean([r["mean_accuracy"] for r in all_results.values()])
    print(f"\n  Overall VGG-Face accuracy : {overall_acc * 100:.2f}%")

    return all_results


if __name__ == "__main__":
    run_experiment()