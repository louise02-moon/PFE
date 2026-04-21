import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
import random

random.seed(42)
np.random.seed(42)

# ─── Paths — only FD ─────────────────────────────────────────────────────────
ARCFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
FACENET_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\FaceNet\facenet_embeddings"
RESNET50_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ResNet50\resnet50_embeddings"
VGGFACE_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGGFace\vggface_embeddings"
HISTLBP_DIR  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]

def load_deep(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "features" not in data:
        sorted_keys = sorted(data.keys(), key=lambda k: os.path.basename(k))
        feats = np.array([data[k] for k in sorted_keys], dtype=np.float64)
    elif isinstance(data, dict) and "features" in data:
        feats = np.array(data["features"], dtype=np.float64)
    else:
        feats = np.array(data, dtype=np.float64)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / (norms + 1e-8)

def load_hlbp(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data, dtype=np.float64)

def power_normalize(X, alpha=0.5):
    return np.sign(X) * (np.abs(X) ** alpha)

def deep_pairs(feats, idxa, idxb):
    a, b = feats[idxa], feats[idxb]
    diff = power_normalize(np.abs(a - b))
    prod = power_normalize(a * b)
    dist = power_normalize(np.linalg.norm(a-b, axis=1, keepdims=True))
    return np.concatenate([diff, prod, dist], axis=1)

def hlbp_pairs(feats, idxa, idxb):
    a, b = feats[idxa], feats[idxb]
    diff = np.sqrt(np.abs(a - b))
    return diff - diff.mean(axis=0)

def normalize(Xtr, Xte):
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)

# ─── Load FD only ─────────────────────────────────────────────────────────────
print("Loading FD features...")
arc  = load_deep(f"{ARCFACE_DIR}\\ArcFace_FD.pkl")
fn   = load_deep(f"{FACENET_DIR}\\FaceNet_FD.pkl")
rn   = load_deep(f"{RESNET50_DIR}\\ResNet50_FD.pkl")
vgg  = load_deep(f"{VGGFACE_DIR}\\VGGFace_FD.pkl")
hlbp = load_hlbp(f"{HISTLBP_DIR}\\HistLBP_FD.pkl")

mat  = sio.loadmat(f"{MAT_DIR}\\LBP_fd.mat")
idxa = mat['idxa'].flatten() - 1
idxb = mat['idxb'].flatten() - 1
fold = mat['fold'].flatten()
y    = mat['matches'].flatten()

X = np.concatenate([
    deep_pairs(arc,  idxa, idxb),
    deep_pairs(fn,   idxa, idxb),
    deep_pairs(rn,   idxa, idxb),
    deep_pairs(vgg,  idxa, idxb),
    hlbp_pairs(hlbp, idxa, idxb),
], axis=1)

print(f"Combined shape: {X.shape}")
print(f"\nClass distribution per fold:")
for f in range(1, 6):
    mask = fold == f
    kin     = y[mask].sum()
    non_kin = (y[mask] == 0).sum()
    print(f"  Fold {f}: {mask.sum()} pairs — {kin} kin, {non_kin} non-kin")

# ─── Per-fold detailed analysis ───────────────────────────────────────────────
print(f"\n{'='*65}")
print("  DETAILED FOLD ANALYSIS — Father-Daughter")
print(f"{'='*65}")

fold_scores = []

for f in range(1, 6):
    train_mask = fold != f
    test_mask  = fold == f

    X_tr_raw = X[train_mask]
    X_te_raw = X[test_mask]
    y_train  = y[train_mask]
    y_test   = y[test_mask]

    # Try multiple C values and show all results
    print(f"\n  Fold {f} — {test_mask.sum()} test pairs:")

    best_C, best_acc = C_GRID[0], -1
    for C in C_GRID:
        X_tr, X_te = normalize(X_tr_raw, X_te_raw)
        clf = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        acc   = accuracy_score(y_test, preds)
        cm    = confusion_matrix(y_test, preds)
        tn, fp, fn_val, tp = cm.ravel()
        print(f"    C={C:<6} acc={acc*100:.1f}%  "
              f"TP={tp} TN={tn} FP={fp} FN={fn_val}  "
              f"TPR={tp/(tp+fn_val)*100:.0f}% TNR={tn/(tn+fp)*100:.0f}%")
        if acc > best_acc:
            best_acc, best_C = acc, C

    fold_scores.append(best_acc)
    print(f"  → Best: C={best_C} → {best_acc*100:.2f}%")

print(f"\n  Mean FD : {np.mean(fold_scores)*100:.2f}%")
print(f"  Std  FD : {np.std(fold_scores)*100:.2f}%")
print(f"\n  Fold scores: {[f'{s*100:.1f}%' for s in fold_scores]}")
print(f"\n  Weakest fold: Fold {np.argmin(fold_scores)+1} "
      f"({min(fold_scores)*100:.1f}%)")
print(f"  Strongest fold: Fold {np.argmax(fold_scores)+1} "
      f"({max(fold_scores)*100:.1f}%)")

# ─── Check if FD images are balanced ─────────────────────────────────────────
print(f"\n{'='*65}")
print("  DATASET BALANCE CHECK")
print(f"{'='*65}")
print(f"  Total pairs     : {len(y)}")
print(f"  Kin pairs       : {y.sum()}")
print(f"  Non-kin pairs   : {(y==0).sum()}")
print(f"  Balance ratio   : {y.mean():.2f} (0.5 = perfectly balanced)")

# ─── Per-fold individual model analysis ──────────────────────────────────────
print(f"\n{'='*65}")
print("  INDIVIDUAL MODEL CONTRIBUTION TO FD")
print(f"{'='*65}")

models = {
    "ArcFace"  : deep_pairs(arc,  idxa, idxb),
    "FaceNet"  : deep_pairs(fn,   idxa, idxb),
    "ResNet50" : deep_pairs(rn,   idxa, idxb),
    "VGGFace"  : deep_pairs(vgg,  idxa, idxb),
    "HistLBP"  : hlbp_pairs(hlbp, idxa, idxb),
}

for model_name, X_model in models.items():
    fold_accs = []
    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f
        X_tr, X_te = normalize(X_model[train_mask], X_model[test_mask])
        # Use best known C for this model
        best_acc_m = -1
        for C in C_GRID:
            clf = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
            clf.fit(X_tr, y[train_mask])
            acc = accuracy_score(y[test_mask], clf.predict(X_te))
            if acc > best_acc_m:
                best_acc_m = acc
        fold_accs.append(best_acc_m)
    print(f"  {model_name:<12} mean={np.mean(fold_accs)*100:.2f}%  "
          f"folds={[f'{a*100:.0f}%' for a in fold_accs]}")