import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import random

# ─── Reproducibility ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
ARCFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
FACENET_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\FaceNet\facenet_embeddings"
VGGFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface_embeddings"
HLBP_DIR     = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FD.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_FD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_FS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_FS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_FS.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_FS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MD.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MD.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MD.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_MD.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "arcface" : f"{ARCFACE_DIR}\\ArcFace_MS.pkl",
        "facenet" : f"{FACENET_DIR}\\FaceNet_MS.pkl",
        "vggface" : f"{VGGFACE_DIR}\\VGGFace_MS.pkl",
        "hlbp"    : f"{HLBP_DIR}\\HistLBP_MS.pkl",
        "mat"     : f"{MAT_DIR}\\LBP_ms.mat",
    },
}

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]

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
def deep_pair_features(feats, idxa, idxb):
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    a, b  = feats[idxa], feats[idxb]
    cos   = np.sum(a * b, axis=1, keepdims=True)
    return np.concatenate([np.abs(a-b), a*b, (a-b)**2, cos], axis=1)

def hlbp_pair_features(feats, idxa, idxb):
    feats = np.sqrt(np.abs(feats))
    a, b  = feats[idxa], feats[idxb]
    diff  = np.abs(a - b)
    mu    = diff.mean(axis=0)
    return diff - mu

# ─── Normalize ────────────────────────────────────────────────────────────────
def normalize(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)

# ─── Train one SVM and return decision scores ─────────────────────────────────
def train_svm_get_scores(X_train, X_test, y_train, C, kernel='rbf'):
    X_tr, X_te = normalize(X_train, X_test)
    clf = SVC(kernel=kernel, C=C, gamma='scale',
              probability=True, random_state=42)
    clf.fit(X_tr, y_train)
    # Return probability of class 1 (kin) as the score
    train_score = clf.predict_proba(X_tr)[:, 1]
    test_score  = clf.predict_proba(X_te)[:, 1]
    return train_score, test_score

# ─── Find best C for one modality ────────────────────────────────────────────
def find_best_C(X_raw, y, fold_ids, c_grid, kernel='rbf'):
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
            scores.append(accuracy_score(yva, clf.predict(Xva_n)))
        mean_s = float(np.mean(scores))
        if mean_s > best_score:
            best_score, best_C = mean_s, C
    return best_C

# ─── Main ─────────────────────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*65}")
    print(f"  Relation : {relation}")
    print(f"{'='*65}")

    arcface_feats = load_embedding(paths["arcface"])
    facenet_feats = load_embedding(paths["facenet"])
    vggface_feats = load_embedding(paths["vggface"])
    hlbp_feats    = load_embedding(paths["hlbp"])

    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    # Build pair features per modality
    arc_X  = deep_pair_features(arcface_feats, idxa, idxb)
    fn_X   = deep_pair_features(facenet_feats, idxa, idxb)
    vgg_X  = deep_pair_features(vggface_feats, idxa, idxb)
    hlbp_X = hlbp_pair_features(hlbp_feats, idxa, idxb)

    fold_scores = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f
        fold_ids   = fold[train_mask]
        y_train    = y[train_mask]
        y_test     = y[test_mask]

        # ── Step 1: find best C per modality via inner CV ─────────────────────
        C_arc  = find_best_C(arc_X[train_mask], y_train, fold_ids, C_GRID, 'rbf')
        C_fn   = find_best_C(fn_X[train_mask], y_train, fold_ids, C_GRID, 'rbf')
        C_vgg  = find_best_C(vgg_X[train_mask], y_train, fold_ids, C_GRID, 'rbf')
        C_hlbp = find_best_C(hlbp_X[train_mask], y_train, fold_ids, C_GRID, 'linear')

        # ── Step 2: train each SVM and get probability scores ─────────────────
        arc_tr,  arc_te  = train_svm_get_scores(arc_X[train_mask],  arc_X[test_mask],  y_train, C_arc,  'rbf')
        fn_tr,   fn_te   = train_svm_get_scores(fn_X[train_mask],   fn_X[test_mask],   y_train, C_fn,   'rbf')
        vgg_tr,  vgg_te  = train_svm_get_scores(vgg_X[train_mask],  vgg_X[test_mask],  y_train, C_vgg,  'rbf')
        hlbp_tr, hlbp_te = train_svm_get_scores(hlbp_X[train_mask], hlbp_X[test_mask], y_train, C_hlbp, 'linear')

        # ── Step 3: stack scores and train meta-classifier ────────────────────
        # Meta features: [arc_score, fn_score, vgg_score, hlbp_score]
        meta_train = np.column_stack([arc_tr, fn_tr, vgg_tr, hlbp_tr])
        meta_test  = np.column_stack([arc_te, fn_te, vgg_te, hlbp_te])

        # Logistic regression learns optimal weight for each model's score
        meta_clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        meta_clf.fit(meta_train, y_train)

        preds = meta_clf.predict(meta_test)
        acc   = accuracy_score(y_test, preds)
        fold_scores.append(acc)

        weights = meta_clf.coef_[0]
        print(f"  Fold {f}: {acc*100:.2f}%  "
              f"weights=[ArcFace:{weights[0]:.2f}, "
              f"FaceNet:{weights[1]:.2f}, "
              f"VGGFace:{weights[2]:.2f}, "
              f"HistLBP:{weights[3]:.2f}]")

    mean_acc = float(np.mean(fold_scores))
    std_acc  = float(np.std(fold_scores))
    all_results[relation] = {
        "fold_scores"   : fold_scores,
        "mean_accuracy" : mean_acc,
        "std_accuracy"  : std_acc,
    }
    print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  FINAL SUMMARY — Score-Level Fusion (Stacking)")
print(f"{'='*65}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")

overall = float(np.mean([r["mean_accuracy"] for r in all_results.values()]))
print(f"\n  Overall Score-Level Fusion : {overall*100:.2f}%")
print(f"  Reference — best so far    : 88.30%")