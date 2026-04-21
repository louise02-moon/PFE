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
VGGFACE_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface_embeddings"
HLBP_DIR    = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_FD.pkl",
        "hlbp":    f"{HLBP_DIR}\\HistLBP_FD.pkl",
        "mat":     f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_FS.pkl",
        "hlbp":    f"{HLBP_DIR}\\HistLBP_FS.pkl",
        "mat":     f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_MD.pkl",
        "hlbp":    f"{HLBP_DIR}\\HistLBP_MD.pkl",
        "mat":     f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "vggface": f"{VGGFACE_DIR}\\VGGFace_MS.pkl",
        "hlbp":    f"{HLBP_DIR}\\HistLBP_MS.pkl",
        "mat":     f"{MAT_DIR}\\LBP_ms.mat",
    },
}

# Grids 
C_GRID      = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
VGG_K_GRID  = [50, 100, 200, 300, 500, 700, 1000]
META_C_GRID = [0.01, 0.1, 1, 10, 100]


# ─── Fisher Score ─────────────────────────────────────────────────────────────
# Computed on RAW features before normalization.
# Ranks each dimension by how well it separates kin from non-kin pairs.
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
    scores  = fisher_score(X_train, y_train)
    top_idx = np.argsort(scores)[::-1][:k]
    return X_train[:, top_idx], X_test[:, top_idx]


# ─── Normalization ────────────────────────────────────────────────────────────
def power_normalize(X_train, X_test):
    """For HLBP — compresses large histogram values."""
    X_train = np.sqrt(np.abs(X_train))
    X_test  = np.sqrt(np.abs(X_test))
    mean_   = X_train.mean(axis=0)
    return X_train - mean_, X_test - mean_


def center(X_train, X_test):
    """For VGG-Face — simple mean subtraction after Fisher selection."""
    mean_ = X_train.mean(axis=0)
    return X_train - mean_, X_test - mean_


# ─── Inner CV: find best C and k for VGG-Face ────────────────────────────────
def find_best_C_and_k(X_raw, y, fold_ids, c_grid, k_grid, norm_fn):
    best_C, best_k, best_score = c_grid[0], k_grid[0], -1
    for k in k_grid:
        for C in c_grid:
            scores = []
            for inner_f in np.unique(fold_ids):
                Xtr = X_raw[fold_ids != inner_f]
                Xte = X_raw[fold_ids == inner_f]
                ytr = y[fold_ids != inner_f]
                yte = y[fold_ids == inner_f]
                Xtr_s, Xte_s = select_top_k(Xtr, Xte, ytr, k)
                Xtr_s, Xte_s = norm_fn(Xtr_s, Xte_s)
                clf = SVC(kernel='linear', C=C, random_state=42)
                clf.fit(Xtr_s, ytr)
                scores.append(accuracy_score(yte, clf.predict(Xte_s)))
            mean_s = np.mean(scores)
            if mean_s > best_score:
                best_score, best_C, best_k = mean_s, C, k
    return best_C, best_k


# ─── Train SVM and return decision scores ────────────────────────────────────
def get_scores(X_tr, X_te, y_tr, norm_fn, C, k):
    X_tr_s, X_te_s = select_top_k(X_tr, X_te, y_tr, k)
    X_tr_s, X_te_s = norm_fn(X_tr_s, X_te_s)
    clf = SVC(kernel='linear', C=C, random_state=42)
    clf.fit(X_tr_s, y_tr)
    return clf.decision_function(X_tr_s), clf.decision_function(X_te_s)


# ─── Inner CV: tune meta-classifier C ────────────────────────────────────────
def find_best_meta_C(meta_train, y_train, fold_ids, c_grid):
    best_C, best_score = c_grid[0], -1
    for C in c_grid:
        scores = []
        for inner_f in np.unique(fold_ids):
            Xtr = meta_train[fold_ids != inner_f]
            Xte = meta_train[fold_ids == inner_f]
            ytr = y_train[fold_ids != inner_f]
            yte = y_train[fold_ids == inner_f]
            clf = LogisticRegression(C=C, random_state=42, max_iter=1000)
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
        print(f"\n{'='*60}")
        print(f"  Relation : {relation}")
        print(f"{'='*60}")

        # ── Load VGG-Face embeddings (plain dict: filename → embedding) ───────
        with open(paths["vggface"], "rb") as f:
            vgg_dict = pickle.load(f)
        sorted_keys = sorted(vgg_dict.keys())
        vgg_feats   = np.array([vgg_dict[k] for k in sorted_keys], dtype=np.float64)

        # L2-normalize
        norms     = np.linalg.norm(vgg_feats, axis=1, keepdims=True)
        vgg_feats = vgg_feats / (norms + 1e-8)

        # ── Load HLBP features (plain numpy array) ────────────────────────────
        with open(paths["hlbp"], "rb") as f:
            hlbp_feats = pickle.load(f)
        hlbp_feats = np.array(hlbp_feats, dtype=np.float64)

        print(f"  VGG-Face shape : {vgg_feats.shape}")
        print(f"  HLBP shape     : {hlbp_feats.shape}")

        assert vgg_feats.shape[0] == hlbp_feats.shape[0], \
            f"Mismatch: VGGFace={vgg_feats.shape[0]}, HLBP={hlbp_feats.shape[0]}"

        # ── Load metadata ─────────────────────────────────────────────────────
        mat     = sio.loadmat(paths["mat"])
        idxa    = mat['idxa'].flatten() - 1
        idxb    = mat['idxb'].flatten() - 1
        fold    = mat['fold'].flatten()
        y       = mat['matches'].flatten()

        # ── Absolute difference pair features ─────────────────────────────────
        X_vgg  = np.abs(vgg_feats[idxa]  - vgg_feats[idxb])
        X_hlbp = np.abs(hlbp_feats[idxa] - hlbp_feats[idxb])

        fold_scores = []

        for f in range(1, 6):
            tr = fold != f
            te = fold == f

            X_vgg_tr,  X_vgg_te  = X_vgg[tr],  X_vgg[te]
            X_hlbp_tr, X_hlbp_te = X_hlbp[tr], X_hlbp[te]
            y_train, y_test       = y[tr], y[te]
            fold_ids              = fold[tr]

            # ── Step 1: tune VGG-Face — Fisher Score + C ──────────────────────
            C_vgg, k_vgg = find_best_C_and_k(
                X_vgg_tr, y_train, fold_ids, C_GRID, VGG_K_GRID, center
            )

            # ── Step 2: tune HLBP — C only, all dims ──────────────────────────
            C_hlbp, _ = find_best_C_and_k(
                X_hlbp_tr, y_train, fold_ids, C_GRID,
                [X_hlbp_tr.shape[1]], power_normalize
            )

            # ── Step 3: get decision scores from both SVMs ────────────────────
            vgg_tr_sc,  vgg_te_sc  = get_scores(
                X_vgg_tr,  X_vgg_te,  y_train, center,          C_vgg,  k_vgg)
            hlbp_tr_sc, hlbp_te_sc = get_scores(
                X_hlbp_tr, X_hlbp_te, y_train, power_normalize, C_hlbp,
                X_hlbp_tr.shape[1])

            # ── Step 4: build and scale meta-features ─────────────────────────
            meta_train = np.column_stack([vgg_tr_sc, hlbp_tr_sc])
            meta_test  = np.column_stack([vgg_te_sc, hlbp_te_sc])

            scaler     = StandardScaler()
            meta_train = scaler.fit_transform(meta_train)
            meta_test  = scaler.transform(meta_test)

            # ── Step 5: tune and train meta-classifier ────────────────────────
            best_meta_C = find_best_meta_C(meta_train, y_train, fold_ids, META_C_GRID)
            meta_clf    = LogisticRegression(C=best_meta_C, random_state=42, max_iter=1000)
            meta_clf.fit(meta_train, y_train)

            preds = meta_clf.predict(meta_test)
            acc   = accuracy_score(y_test, preds)
            fold_scores.append(acc)

            w_vgg, w_hlbp = meta_clf.coef_[0]
            print(f"  Fold {f}: {acc*100:.2f}%  "
                  f"(C_vgg={C_vgg}, k={k_vgg}, "
                  f"C_hlbp={C_hlbp}, meta_C={best_meta_C}, "
                  f"w_vgg={w_vgg:.3f}, w_hlbp={w_hlbp:.3f})")

        mean_acc = np.mean(fold_scores)
        std_acc  = np.std(fold_scores)
        all_results[relation] = {
            "fold_scores":   fold_scores,
            "mean_accuracy": mean_acc,
            "std_accuracy":  std_acc,
        }
        print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — VGGFace (Fisher Score) + HLBP Stacking")
    print(f"{'='*60}")
    print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
    print("-" * 44)
    for relation, res in all_results.items():
        print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
              f" {res['std_accuracy']*100:>7.2f}%")

    overall = np.mean([r["mean_accuracy"] for r in all_results.values()])
    print(f"\n  Overall accuracy : {overall*100:.2f}%")
    return all_results


if __name__ == "__main__":
    run_experiment()