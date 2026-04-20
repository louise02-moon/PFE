import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random

# ─── Reproducibility ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
RESNET101_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ResNet101\resnet101_embeddings"
MAT_DIR       = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "resnet101": f"{RESNET101_DIR}\\ResNet101_FD.pkl",
        "mat":       f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "resnet101": f"{RESNET101_DIR}\\ResNet101_FS.pkl",
        "mat":       f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "resnet101": f"{RESNET101_DIR}\\ResNet101_MD.pkl",
        "mat":       f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "resnet101": f"{RESNET101_DIR}\\ResNet101_MS.pkl",
        "mat":       f"{MAT_DIR}\\LBP_ms.mat",
    },
}

# ─── Per-relation configuration ───────────────────────────────────────────────
RELATION_CONFIG = {
    "Father-Daughter": {
        "kernel":       "rbf",
        "C_grid":       [0.001, 0.01, 0.1, 1, 10, 100],
        "deep_weight":  0.7,
        "fusion_modes": ["diff", "prod", "sq_diff"],
    },
    "Father-Son": {
        "kernel":       "rbf",
        "C_grid":       [0.001, 0.01, 0.1, 1, 10, 100],
        "deep_weight":  1.0,
        "fusion_modes": ["diff", "prod", "sq_diff", "cosine"],
    },
    "Mother-Daughter": {
        "kernel":       "rbf",
        "C_grid":       [0.001, 0.01, 0.1, 1, 10, 100],
        "deep_weight":  1.0,
        "fusion_modes": ["diff", "prod", "sq_diff", "cosine"],
    },
    "Mother-Son": {
        "kernel":       "rbf",
        "C_grid":       [0.001, 0.01, 0.1, 1, 10, 100],
        "deep_weight":  0.8,
        "fusion_modes": ["diff", "prod", "sq_diff", "cosine"],
    },
}

# ─── Load embedding ───────────────────────────────────────────────────────────
def load_embedding(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "features" not in data:
        sorted_keys = sorted(data.keys())
        feats = np.array([data[k] for k in sorted_keys], dtype=np.float64)
    elif isinstance(data, dict) and "features" in data:
        feats = np.array(data["features"], dtype=np.float64)
    else:
        feats = np.array(data, dtype=np.float64)

    return feats

# ─── Build pair features ──────────────────────────────────────────────────────
def deep_pair_features(feats, idxa, idxb, fusion_modes, weight=1.0):
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    a, b  = feats[idxa], feats[idxb]

    parts = []
    if "diff" in fusion_modes:
        parts.append(np.abs(a - b))
    if "prod" in fusion_modes:
        parts.append(a * b)
    if "sq_diff" in fusion_modes:
        parts.append((a - b) ** 2)
    if "cosine" in fusion_modes:
        parts.append(np.sum(a * b, axis=1, keepdims=True))

    result = np.concatenate(parts, axis=1)
    return result * weight

# ─── Normalization ────────────────────────────────────────────────────────────
def normalize(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)

# ─── Inner CV for best C ──────────────────────────────────────────────────────
def find_best_C(X_raw, y, fold_ids, c_grid, kernel):
    best_C, best_score = c_grid[0], -1.0

    for C in c_grid:
        scores = []
        for inner_f in np.unique(fold_ids):
            Xtr = X_raw[fold_ids != inner_f]
            Xva = X_raw[fold_ids == inner_f]
            ytr = y[fold_ids != inner_f]
            yva = y[fold_ids == inner_f]

            Xtr_n, Xva_n = normalize(Xtr, Xva)

            clf = SVC(kernel=kernel, C=C, gamma='scale', random_state=42)
            clf.fit(Xtr_n, ytr)

            preds = clf.predict(Xva_n)
            scores.append(accuracy_score(yva, preds))

        mean_s = float(np.mean(scores))
        if mean_s > best_score:
            best_score, best_C = mean_s, C

    return best_C

# ─── Main ─────────────────────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    cfg = RELATION_CONFIG[relation]

    print(f"\n{'='*65}")
    print(f"  Relation : {relation}")
    print(f"  Config   : kernel={cfg['kernel']}, "
          f"deep_w={cfg['deep_weight']}, "
          f"fusion={cfg['fusion_modes']}")
    print(f"{'='*65}")

    resnet101_feats = load_embedding(paths["resnet101"])

    mat  = sio.loadmat(paths["mat"])
    idxa = mat["idxa"].flatten() - 1
    idxb = mat["idxb"].flatten() - 1
    fold = mat["fold"].flatten()
    y    = mat["matches"].flatten()

    X = deep_pair_features(
        resnet101_feats,
        idxa,
        idxb,
        cfg["fusion_modes"],
        cfg["deep_weight"]
    )

    print(f"  Feature shape : {X.shape}")

    fold_scores = []
    best_Cs = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y[train_mask]
        y_test      = y[test_mask]

        best_C = find_best_C(
            X_train_raw,
            y_train,
            fold[train_mask],
            cfg["C_grid"],
            cfg["kernel"]
        )
        best_Cs.append(best_C)

        X_train, X_test = normalize(X_train_raw, X_test_raw)

        clf = SVC(
            kernel=cfg["kernel"],
            C=best_C,
            gamma='scale',
            random_state=42
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        fold_scores.append(acc)

        print(f"  Fold {f}: {acc*100:.2f}%   (C={best_C})")

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))

    all_results[relation] = {
        "fold_scores":   fold_scores,
        "mean_accuracy": mean_acc,
        "std_accuracy":  std_acc,
        "best_Cs":       best_Cs,
    }

    print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  FINAL SUMMARY — ResNet101 Alone")
print(f"{'='*65}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)

for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}% "
          f"{res['std_accuracy']*100:>7.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall ResNet101 Alone : {overall*100:.2f}%")