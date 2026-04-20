import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC
import random

# Reproductibilité
random.seed(42)
np.random.seed(42)

# ─── Chemins des fichiers ───────────────────────────────────────────────────
PKL_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Color LBP\RGB\Color_LBP_RGB_feature_vectors"
MAT_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "pkl": f"{PKL_DIR}\\ColorLBP_RGB_FD.pkl",
        "mat": f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "pkl": f"{PKL_DIR}\\ColorLBP_RGB_FS.pkl",
        "mat": f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "pkl": f"{PKL_DIR}\\ColorLBP_RGB_MD.pkl",
        "mat": f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "pkl": f"{PKL_DIR}\\ColorLBP_RGB_MS.pkl",
        "mat": f"{MAT_DIR}\\LBP_ms.mat",
    },
}

# ─── Boucle principale sur les 4 relations ──────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*50}")
    print(f"  Relation : {relation}")
    print(f"{'='*50}")

    # Chargement des features
    with open(paths["pkl"], "rb") as f:
        ux = pickle.load(f)

    ux = np.array(ux)

    # Chargement du fichier .mat
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    matches = mat['matches'].flatten()

    # Construction des paires
    X = np.abs(ux[idxa] - ux[idxb])
    y = matches

    # 5-Fold Cross-Validation
    fold_scores = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_train = X[train_mask]
        X_test  = X[test_mask]
        y_train = y[train_mask]
        y_test  = y[test_mask]

        # ─── Power normalization ────────────────────────────────────────────
        scaler = PowerTransformer(method='yeo-johnson')

        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # ─── SVM classifier ────────────────────────────────────────────────
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        fold_scores.append(acc)

        print(f"  Fold {f}: Accuracy = {acc * 100:.2f}%")

    mean_acc = np.mean(fold_scores)
    all_results[relation] = {
        "fold_scores": fold_scores,
        "mean_accuracy": mean_acc
    }

    print(f"  Mean Accuracy : {mean_acc * 100:.2f}%")

# ─── Récapitulatif final ─────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("  RÉCAPITULATIF FINAL")
print(f"{'='*50}")
print(f"{'Relation':<20} {'Mean Accuracy':>15}")
print("-" * 36)

for relation, res in all_results.items():
    print(f"{relation:<20} {res['mean_accuracy'] * 100:>14.2f}%")

# ─── Accuracy globale Color LBP RGB ─────────────────────────────────────────
overall_acc = np.mean([res['mean_accuracy'] for res in all_results.values()])

print(f"\n{'='*50}")
print(f"  ACCURACY GLOBALE — COLOR LBP RGB + SVM (POWER)")
print(f"{'='*50}")
print(f"  (Mean Accuracy FD + FS + MD + MS) / 4")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────")
print(f"  Color LBP RGB Accuracy : {overall_acc * 100:.2f}%")