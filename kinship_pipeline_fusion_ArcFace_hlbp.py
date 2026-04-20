import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random

# ─── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── File paths ───────────────────────────────────────────────────────────────
ARCFACE_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
HLBP_DIR    = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "arcface": f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "hlbp":    f"{HLBP_DIR}\\HistLBP_FD.pkl",
        "mat":     f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "arcface": f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "hlbp":    f"{HLBP_DIR}\\HistLBP_FS.pkl",
        "mat":     f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "arcface": f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "hlbp":    f"{HLBP_DIR}\\HistLBP_MD.pkl",
        "mat":     f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "arcface": f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "hlbp":    f"{HLBP_DIR}\\HistLBP_MS.pkl",
        "mat":     f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID = [0.01, 0.1, 1, 10, 100, 1000]


# ─── Normalization ────────────────────────────────────────────────────────────
def power_normalize(X_train, X_test):
    X_train = np.sqrt(np.abs(X_train))
    X_test  = np.sqrt(np.abs(X_test))
    mean_   = X_train.mean(axis=0)
    return X_train - mean_, X_test - mean_

def center(X_train, X_test):
    mean_ = X_train.mean(axis=0)
    return X_train - mean_, X_test - mean_


# ─── Find best C via inner CV ─────────────────────────────────────────────────
def find_best_C(X_raw, y, fold_ids, c_grid, norm_fn):
    best_C, best_score = c_grid[0], -1
    for C in c_grid:
        scores = []
        for inner_f in np.unique(fold_ids):
            Xtr, Xte = X_raw[fold_ids != inner_f], X_raw[fold_ids == inner_f]
            ytr, yte = y[fold_ids != inner_f],     y[fold_ids == inner_f]
            Xtr, Xte = norm_fn(Xtr, Xte)
            clf = SVC(kernel='linear', C=C, random_state=42)
            clf.fit(Xtr, ytr)
            scores.append(accuracy_score(yte, clf.predict(Xte)))
        mean_s = np.mean(scores)
        if mean_s > best_score:
            best_score, best_C = mean_s, C
    return best_C


# ─── Train SVM, return decision scores for train AND test ────────────────────
def get_decision_scores(X_train_raw, X_test_raw, y_train, norm_fn, C):
    Xtr, Xte = norm_fn(X_train_raw, X_test_raw)
    clf = SVC(kernel='linear', C=C, random_state=42)
    clf.fit(Xtr, y_train)
    return clf.decision_function(Xtr), clf.decision_function(Xte)


# ─── Main loop ────────────────────────────────────────────────────────────────
def run_experiment():
    all_results = {}

    for relation, paths in RELATIONS.items():
        print(f"\n{'='*55}")
        print(f"  Relation : {relation}")
        print(f"{'='*55}")

        # ── Load ArcFace ──────────────────────────────────────────────────────
        with open(paths["arcface"], "rb") as f:
            arc_dict = pickle.load(f)
        sorted_keys = sorted(arc_dict.keys())
        arc_feats   = np.array([arc_dict[k] for k in sorted_keys], dtype=np.float64)
        norms       = np.linalg.norm(arc_feats, axis=1, keepdims=True)
        arc_feats   = arc_feats / (norms + 1e-8)

        # ── Load HLBP ─────────────────────────────────────────────────────────
        with open(paths["hlbp"], "rb") as f:
            hlbp_feats = pickle.load(f)
        hlbp_feats = np.array(hlbp_feats, dtype=np.float64)

        assert arc_feats.shape[0] == hlbp_feats.shape[0], \
            f"Mismatch: ArcFace={arc_feats.shape[0]}, HLBP={hlbp_feats.shape[0]}"

        # ── Load metadata ─────────────────────────────────────────────────────
        mat     = sio.loadmat(paths["mat"])
        idxa    = mat['idxa'].flatten() - 1
        idxb    = mat['idxb'].flatten() - 1
        fold    = mat['fold'].flatten()
        y       = mat['matches'].flatten()

        # ── Pair difference features ──────────────────────────────────────────
        X_arc  = np.abs(arc_feats[idxa]  - arc_feats[idxb])
        X_hlbp = np.abs(hlbp_feats[idxa] - hlbp_feats[idxb])

        fold_scores = []

        for f in range(1, 6):
            tr = fold != f
            te = fold == f

            X_arc_tr,  X_arc_te  = X_arc[tr],  X_arc[te]
            X_hlbp_tr, X_hlbp_te = X_hlbp[tr], X_hlbp[te]
            y_train, y_test       = y[tr], y[te]
            fold_ids              = fold[tr]

            # ── Step 1: find best C for each base SVM ─────────────────────────
            C_arc  = find_best_C(X_arc_tr,  y_train, fold_ids, C_GRID, center)
            C_hlbp = find_best_C(X_hlbp_tr, y_train, fold_ids, C_GRID, power_normalize)

            # ── Step 2: get decision scores from both SVMs ────────────────────
            arc_tr_scores,  arc_te_scores  = get_decision_scores(
                X_arc_tr,  X_arc_te,  y_train, center,          C_arc)
            hlbp_tr_scores, hlbp_te_scores = get_decision_scores(
                X_hlbp_tr, X_hlbp_te, y_train, power_normalize, C_hlbp)

            # ── Step 3: stack scores into meta-features ───────────────────────
            # Shape: (n_train, 2) and (n_test, 2)
            # Each row = [arcface_score, hlbp_score] for one pair
            meta_train = np.column_stack([arc_tr_scores,  hlbp_tr_scores])
            meta_test  = np.column_stack([arc_te_scores,  hlbp_te_scores])

            # ── Step 4: scale meta-features ───────────────────────────────────
            # Decision scores can have very different ranges across SVMs.
            # StandardScaler brings them to the same scale before LR sees them.
            scaler     = StandardScaler()
            meta_train = scaler.fit_transform(meta_train)
            meta_test  = scaler.transform(meta_test)

            # ── Step 5: train logistic regression meta-classifier ─────────────
            # It learns the optimal weight for each SVM's score,
            # and crucially, can learn non-linear combinations too (e.g.
            # "trust HLBP when ArcFace is uncertain").
            meta_clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
            meta_clf.fit(meta_train, y_train)

            preds = meta_clf.predict(meta_test)
            acc   = accuracy_score(y_test, preds)
            fold_scores.append(acc)

            # Show learned weights so you can see how much each model contributes
            w_arc, w_hlbp = meta_clf.coef_[0]
            print(f"  Fold {f}: {acc*100:.2f}%  "
                  f"(C_arc={C_arc}, C_hlbp={C_hlbp}, "
                  f"w_arc={w_arc:.3f}, w_hlbp={w_hlbp:.3f})")

        mean_acc = np.mean(fold_scores)
        std_acc  = np.std(fold_scores)
        all_results[relation] = {
            "fold_scores":   fold_scores,
            "mean_accuracy": mean_acc,
            "std_accuracy":  std_acc,
        }
        print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  FINAL SUMMARY — Stacking Fusion (ArcFace + HLBP)")
    print(f"{'='*55}")
    print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
    print("-" * 44)
    for relation, res in all_results.items():
        print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
              f" {res['std_accuracy']*100:>7.2f}%")

    overall = np.mean([r["mean_accuracy"] for r in all_results.values()])
    print(f"\n  Overall stacking accuracy : {overall*100:.2f}%")
    return all_results


if __name__ == "__main__":
    run_experiment()