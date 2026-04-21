import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import random

# ─── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
# Point this to whichever layer embeddings you want to test.
# Recommended: try pool5 first (512-dim, more general than fc layers).
# If you only have fc7 embeddings, set LAYER_NAME = "fc7_relu" and it still works.
VGGFACE_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface_pool5_embeddings"
MAT_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

LAYER_NAME  = "pool5"

RELATIONS = {
    "Father-Daughter": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_FD_pool5.pkl",
        "mat":     f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_FS_pool5.pkl",
        "mat":     f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_MD_pool5.pkl",
        "mat":     f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_MS_pool5.pkl",
        "mat":     f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID     = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
TOP_K_GRID = [100, 200, 300, 500, 700, 850, 1000, 1250, 1500]


# ─── Fisher Score ─────────────────────────────────────────────────────────────
# MUST be computed on raw (or power-normalized) features,
# BEFORE mean centering — centering doesn't change Fisher scores
# but StandardScaler (dividing by std) distorts them.
def fisher_score(X, y):
    scores = np.zeros(X.shape[1], dtype=np.float64)
    for c in np.unique(y):
        Xc    = X[y == c]
        Xnot  = X[y != c]
        num   = (Xc.mean(axis=0) - Xnot.mean(axis=0)) ** 2
        denom = Xc.var(axis=0) + Xnot.var(axis=0) + 1e-8
        scores += num / denom
    return scores


def select_top_k(X_train, X_test, y_train, k):
    # Fisher Score on raw training features — before any normalization
    scores  = fisher_score(X_train, y_train)
    top_idx = np.argsort(scores)[::-1][:k]
    return X_train[:, top_idx], X_test[:, top_idx]


# ─── Power normalization ──────────────────────────────────────────────────────
# Applied AFTER Fisher Score selection.
# sqrt(|x|) compresses large activations common in fc layer outputs.
def power_normalize(X_train, X_test):
    X_train = np.sqrt(np.abs(X_train))
    X_test  = np.sqrt(np.abs(X_test))
    mean_   = X_train.mean(axis=0)
    return X_train - mean_, X_test - mean_


# ─── Inner CV: find best C and k jointly ─────────────────────────────────────
def find_best_C_and_k(X_raw, y, fold_ids, c_grid, k_grid):
    best_C, best_k, best_score = c_grid[0], k_grid[0], -1.0
    for k in k_grid:
        for C in c_grid:
            scores = []
            for inner_f in np.unique(fold_ids):
                Xtr = X_raw[fold_ids != inner_f]
                Xva = X_raw[fold_ids == inner_f]
                ytr = y[fold_ids != inner_f]
                yva = y[fold_ids == inner_f]

                # Step 1: Fisher Score selection (on raw features)
                Xtr_s, Xva_s = select_top_k(Xtr, Xva, ytr, k)

                # Step 2: Power normalize AFTER selection
                Xtr_s, Xva_s = power_normalize(Xtr_s, Xva_s)

                clf = SVC(kernel="linear", C=C, random_state=42)
                clf.fit(Xtr_s, ytr)
                scores.append(accuracy_score(yva, clf.predict(Xva_s)))

            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score, best_C, best_k = mean_s, C, k
    return best_C, best_k


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_experiment():
    all_results = {}

    for relation, paths in RELATIONS.items():
        print(f"\n{'='*65}")
        print(f"  Relation : {relation}")
        print(f"{'='*65}")

        with open(paths["vggface"], "rb") as f:
            payload = pickle.load(f)

        # payload is a dict with keys: features, filenames, layer, etc.
        feats     = np.array(payload["features"],  dtype=np.float64)
        filenames = payload["filenames"]

        print(f"  Layer         : {payload.get('layer', 'unknown')}")
        print(f"  Feature shape : {feats.shape}")
        print(f"  Feature range : [{feats.min():.4f}, {feats.max():.4f}]")

        # L2-normalize each embedding (unit vector)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / (norms + 1e-8)

        mat  = sio.loadmat(paths["mat"])
        idxa = mat["idxa"].flatten() - 1
        idxb = mat["idxb"].flatten() - 1
        fold = mat["fold"].flatten()
        y    = mat["matches"].flatten()

        # Safety: make sure filenames align with .mat indices
        assert feats.shape[0] >= max(idxa.max(), idxb.max()) + 1, \
            f"Index out of range: {feats.shape[0]} embeddings but mat needs index {max(idxa.max(), idxb.max())}"

        # Absolute difference pair features
        X = np.abs(feats[idxa] - feats[idxb])
        print(f"  Pair features  : {X.shape}")

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

            # Find best C and k via inner CV
            best_C, best_k = find_best_C_and_k(
                X_train_raw, y_train, fold_ids, C_GRID, TOP_K_GRID
            )
            chosen_Cs.append(best_C)
            chosen_ks.append(best_k)

            # Final train/test with best hyperparams
            # Step 1: Fisher Score selection
            X_train_s, X_test_s = select_top_k(X_train_raw, X_test_raw, y_train, best_k)

            # Step 2: Power normalize
            X_train_s, X_test_s = power_normalize(X_train_s, X_test_s)

            clf = SVC(kernel="linear", C=best_C, random_state=42)
            clf.fit(X_train_s, y_train)

            acc = accuracy_score(y_test, clf.predict(X_test_s))
            fold_scores.append(acc)
            print(f"  Fold {f}: {acc*100:.2f}%   (C={best_C}, k={best_k})")

        mean_acc = float(np.mean(fold_scores))
        std_acc  = float(np.std(fold_scores))

        all_results[relation] = {
            "fold_scores":   fold_scores,
            "mean_accuracy": mean_acc,
            "std_accuracy":  std_acc,
            "best_Cs":       chosen_Cs,
            "best_ks":       chosen_ks,
        }
        print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY — VGG-Face + Fisher Score + Linear SVM")
    print(f"{'='*65}")
    print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
    print("-" * 44)
    for relation, res in all_results.items():
        print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
              f" {res['std_accuracy']*100:>7.2f}%")

    overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
    print(f"\n  Overall VGG-Face + Fisher + SVM : {overall*100:.2f}%")
    print(f"  Reference — paper target        : ~84.00%")
    return all_results


if __name__ == "__main__":
    run_experiment()