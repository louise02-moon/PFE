import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random
import os

# ─── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── File paths ───────────────────────────────────────────────────────────────
RESNET_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface2_resnet50_embeddings"
HLBP_DIR    = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "resnet": os.path.join(RESNET_DIR, "VGGFace2_FD_resnet50.pkl"),
        "hlbp":   os.path.join(HLBP_DIR,  "HistLBP_FD.pkl"),
        "mat":    os.path.join(MAT_DIR,    "LBP_fd.mat"),
    },
    "Father-Son": {
        "resnet": os.path.join(RESNET_DIR, "VGGFace2_FS_resnet50.pkl"),
        "hlbp":   os.path.join(HLBP_DIR,  "HistLBP_FS.pkl"),
        "mat":    os.path.join(MAT_DIR,    "LBP_fs.mat"),
    },
    "Mother-Daughter": {
        "resnet": os.path.join(RESNET_DIR, "VGGFace2_MD_resnet50.pkl"),
        "hlbp":   os.path.join(HLBP_DIR,  "HistLBP_MD.pkl"),
        "mat":    os.path.join(MAT_DIR,    "LBP_md.mat"),
    },
    "Mother-Son": {
        "resnet": os.path.join(RESNET_DIR, "VGGFace2_MS_resnet50.pkl"),
        "hlbp":   os.path.join(HLBP_DIR,  "HistLBP_MS.pkl"),
        "mat":    os.path.join(MAT_DIR,    "LBP_ms.mat"),
    },
}

# ResNet50 is 2048-dim — search up to 2048
RESNET_C_GRID  = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
RESNET_K_GRID  = [50, 100, 200, 300, 500, 700, 1000, 1500, 2048]

# HLBP: C only, all dims (proven best)
HLBP_C_GRID    = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# Meta-classifier
META_C_GRID    = [0.01, 0.1, 1, 10, 100]


# ─── Fisher Score ─────────────────────────────────────────────────────────────
# Computed on RAW features before normalization.
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
    idx = np.argsort(fisher_score(X_train, y_train))[::-1][:k]
    return X_train[:, idx], X_test[:, idx]


# ─── Normalization ────────────────────────────────────────────────────────────
def power_normalize(X_train, X_test):
    """For HLBP and ResNet50 — compresses large activations."""
    X_train = np.sqrt(np.abs(X_train))
    X_test  = np.sqrt(np.abs(X_test))
    mean_   = X_train.mean(axis=0)
    return X_train - mean_, X_test - mean_


# ─── Inner CV: find best C and k for ResNet50 ────────────────────────────────
def find_best_resnet_params(X_raw, y, fold_ids, c_grid, k_grid):
    best_C, best_k, best_score = c_grid[0], k_grid[0], -1.0
    for k in k_grid:
        for C in c_grid:
            scores = []
            for inner_f in np.unique(fold_ids):
                Xtr = X_raw[fold_ids != inner_f]
                Xva = X_raw[fold_ids == inner_f]
                ytr = y[fold_ids != inner_f]
                yva = y[fold_ids == inner_f]
                # Fisher Score FIRST (on raw), then power normalize
                Xtr_s, Xva_s = select_top_k(Xtr, Xva, ytr, k)
                Xtr_s, Xva_s = power_normalize(Xtr_s, Xva_s)
                clf = SVC(kernel="linear", C=C, random_state=42)
                clf.fit(Xtr_s, ytr)
                scores.append(accuracy_score(yva, clf.predict(Xva_s)))
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score, best_C, best_k = mean_s, C, k
    return best_C, best_k


# ─── Inner CV: find best C for HLBP ──────────────────────────────────────────
def find_best_hlbp_C(X_raw, y, fold_ids, c_grid):
    best_C, best_score = c_grid[0], -1.0
    for C in c_grid:
        scores = []
        for inner_f in np.unique(fold_ids):
            Xtr = X_raw[fold_ids != inner_f]
            Xva = X_raw[fold_ids == inner_f]
            ytr = y[fold_ids != inner_f]
            yva = y[fold_ids == inner_f]
            Xtr_n, Xva_n = power_normalize(Xtr, Xva)
            clf = SVC(kernel="linear", C=C, random_state=42)
            clf.fit(Xtr_n, ytr)
            scores.append(accuracy_score(yva, clf.predict(Xva_n)))
        mean_s = float(np.mean(scores))
        if mean_s > best_score:
            best_score, best_C = mean_s, C
    return best_C


# ─── Train SVM and return decision scores ────────────────────────────────────
def get_resnet_scores(X_tr, X_te, y_tr, C, k):
    X_tr_s, X_te_s = select_top_k(X_tr, X_te, y_tr, k)
    X_tr_s, X_te_s = power_normalize(X_tr_s, X_te_s)
    clf = SVC(kernel="linear", C=C, random_state=42)
    clf.fit(X_tr_s, y_tr)
    return clf.decision_function(X_tr_s), clf.decision_function(X_te_s)


def get_hlbp_scores(X_tr, X_te, y_tr, C):
    X_tr_n, X_te_n = power_normalize(X_tr, X_te)
    clf = SVC(kernel="linear", C=C, random_state=42)
    clf.fit(X_tr_n, y_tr)
    return clf.decision_function(X_tr_n), clf.decision_function(X_te_n)


# ─── Inner CV: tune meta-classifier C ────────────────────────────────────────
def find_best_meta_C(meta_train, y_train, fold_ids, c_grid):
    best_C, best_score = c_grid[0], -1.0
    for C in c_grid:
        scores = []
        for inner_f in np.unique(fold_ids):
            Xtr = meta_train[fold_ids != inner_f]
            Xva = meta_train[fold_ids == inner_f]
            ytr = y_train[fold_ids != inner_f]
            yva = y_train[fold_ids == inner_f]
            clf = LogisticRegression(C=C, random_state=42, max_iter=1000)
            clf.fit(Xtr, ytr)
            scores.append(accuracy_score(yva, clf.predict(Xva)))
        mean_s = float(np.mean(scores))
        if mean_s > best_score:
            best_score, best_C = mean_s, C
    return best_C


# ─── Main loop ────────────────────────────────────────────────────────────────
def run_experiment():
    all_results = {}

    for relation, paths in RELATIONS.items():
        print(f"\n{'='*65}")
        print(f"  Relation : {relation}")
        print(f"{'='*65}")

        # ── Load ResNet50 embeddings ──────────────────────────────────────────
        with open(paths["resnet"], "rb") as f:
            payload = pickle.load(f)

        feats     = np.array(payload["features"], dtype=np.float64)
        filenames = payload["filenames"]

        print(f"  Layer         : {payload.get('layer', 'resnet50')}")
        print(f"  ResNet shape  : {feats.shape}")
        print(f"  Feature range : [{feats.min():.4f}, {feats.max():.4f}]")

        # L2-normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / (norms + 1e-8)

        # ── Load HLBP features ────────────────────────────────────────────────
        with open(paths["hlbp"], "rb") as f:
            hlbp_feats = pickle.load(f)
        hlbp_feats = np.array(hlbp_feats, dtype=np.float64)
        print(f"  HLBP shape    : {hlbp_feats.shape}")

        assert feats.shape[0] == hlbp_feats.shape[0], \
            f"Mismatch: ResNet={feats.shape[0]}, HLBP={hlbp_feats.shape[0]}"

        # ── Load metadata ─────────────────────────────────────────────────────
        mat  = sio.loadmat(paths["mat"])
        idxa = mat["idxa"].flatten() - 1
        idxb = mat["idxb"].flatten() - 1
        fold = mat["fold"].flatten()
        y    = mat["matches"].flatten()

        # ── Pair features ─────────────────────────────────────────────────────
        X_resnet = np.abs(feats[idxa]      - feats[idxb])
        X_hlbp   = np.abs(hlbp_feats[idxa] - hlbp_feats[idxb])

        fold_scores = []

        for f in range(1, 6):
            tr = fold != f
            te = fold == f

            X_res_tr,  X_res_te  = X_resnet[tr], X_resnet[te]
            X_hlbp_tr, X_hlbp_te = X_hlbp[tr],   X_hlbp[te]
            y_train, y_test       = y[tr],         y[te]
            fold_ids              = fold[tr]

            # ── Tune ResNet50 — Fisher Score + C ──────────────────────────────
            C_res, k_res = find_best_resnet_params(
                X_res_tr, y_train, fold_ids, RESNET_C_GRID, RESNET_K_GRID
            )

            # ── Tune HLBP — C only, all dims ──────────────────────────────────
            C_hlbp = find_best_hlbp_C(
                X_hlbp_tr, y_train, fold_ids, HLBP_C_GRID
            )

            # ── Get decision scores ───────────────────────────────────────────
            res_tr_sc,  res_te_sc  = get_resnet_scores(
                X_res_tr,  X_res_te,  y_train, C_res,  k_res)
            hlbp_tr_sc, hlbp_te_sc = get_hlbp_scores(
                X_hlbp_tr, X_hlbp_te, y_train, C_hlbp)

            # ── Stacking ──────────────────────────────────────────────────────
            meta_train = np.column_stack([res_tr_sc,  hlbp_tr_sc])
            meta_test  = np.column_stack([res_te_sc,  hlbp_te_sc])

            scaler     = StandardScaler()
            meta_train = scaler.fit_transform(meta_train)
            meta_test  = scaler.transform(meta_test)

            best_meta_C = find_best_meta_C(meta_train, y_train, fold_ids, META_C_GRID)
            meta_clf    = LogisticRegression(C=best_meta_C, random_state=42, max_iter=1000)
            meta_clf.fit(meta_train, y_train)

            preds = meta_clf.predict(meta_test)
            acc   = accuracy_score(y_test, preds)
            fold_scores.append(acc)

            w_res, w_hlbp = meta_clf.coef_[0]
            print(f"  Fold {f}: {acc*100:.2f}%  "
                  f"(C_res={C_res}, k={k_res}, "
                  f"C_hlbp={C_hlbp}, meta_C={best_meta_C}, "
                  f"w_res={w_res:.3f}, w_hlbp={w_hlbp:.3f})")

        mean_acc = float(np.mean(fold_scores))
        std_acc  = float(np.std(fold_scores))
        all_results[relation] = {
            "fold_scores":   fold_scores,
            "mean_accuracy": mean_acc,
            "std_accuracy":  std_acc,
        }
        print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY — ResNet50 (Fisher Score) + HLBP Stacking")
    print(f"{'='*65}")
    print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
    print("-" * 44)
    for relation, res in all_results.items():
        print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
              f" {res['std_accuracy']*100:>7.2f}%")

    overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
    print(f"\n  Overall accuracy         : {overall*100:.2f}%")
    print(f"  Previous best (VGGFace)  : 88.35%")
    print(f"  Target                   : 91.00%")
    return all_results


if __name__ == "__main__":
    run_experiment()