import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import random
 
# Reproductibilité
random.seed(42)
np.random.seed(42)
 
# ─── Chemins des fichiers ────────────────────────────────────────────────────
PKL_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors"
MAT_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"
 
RELATIONS = {
    "Father-Daughter": {
        "pkl": f"{PKL_DIR}\\Color_HLBP_RGB_FD.pkl",
        "mat": f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "pkl": f"{PKL_DIR}\\Color_HLBP_RGB_FS.pkl",
        "mat": f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "pkl": f"{PKL_DIR}\\Color_HLBP_RGB_MD.pkl",
        "mat": f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "pkl": f"{PKL_DIR}\\Color_HLBP_RGB_MS.pkl",
        "mat": f"{MAT_DIR}\\LBP_ms.mat",
    },
}
 
C_GRID = [0.01, 0.1, 1, 10, 100, 1000]
 
# ─── Transformations ─────────────────────────────────────────────────────────
def sqrt_transform(X):
    return np.sqrt(np.clip(X, 0, None))
 
# ─── Sélection de C par CV interne (4 folds d'entraînement) ─────────────────
def find_best_C(X_train_raw, y_train, train_fold_ids, c_grid):
    inner_folds = np.unique(train_fold_ids)
    best_C, best_score = c_grid[0], -1
 
    for C in c_grid:
        scores = []
        for inner_f in inner_folds:
            imask_tr = train_fold_ids != inner_f
            imask_te = train_fold_ids == inner_f
 
            Xtr, Xte = X_train_raw[imask_tr], X_train_raw[imask_te]
            ytr, yte = y_train[imask_tr],     y_train[imask_te]
 
            Xtr = sqrt_transform(Xtr)
            Xte = sqrt_transform(Xte)
 
            sc  = MinMaxScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
 
            clf = SVC(kernel="rbf", C=C, gamma="scale", random_state=42)
            clf.fit(Xtr, ytr)
            scores.append(accuracy_score(yte, clf.predict(Xte)))
 
        mean_s = np.mean(scores)
        if mean_s > best_score:
            best_score, best_C = mean_s, C
 
    return best_C
 
# ─── Boucle principale ───────────────────────────────────────────────────────
all_results = {}
 
for relation, paths in RELATIONS.items():
    print(f"\n{'='*55}")
    print(f"  Relation : {relation}")
    print(f"{'='*55}")
 
    with open(paths["pkl"], "rb") as f:
        ux = pickle.load(f)
    ux = np.array(ux, dtype=np.float64)
 
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat["idxa"].flatten() - 1
    idxb    = mat["idxb"].flatten() - 1
    fold    = mat["fold"].flatten()
    matches = mat["matches"].flatten()
 
    # Différence absolue — fusion simple, adaptée aux petits datasets
    X = np.abs(ux[idxa] - ux[idxb])
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
 
        # 1. Sqrt transform
        X_train = sqrt_transform(X_train_raw)
        X_test  = sqrt_transform(X_test_raw)
 
        # 2. MinMaxScaler — fit sur train uniquement
        scaler  = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
 
        # 3. Sélection de C par CV interne
        best_C = find_best_C(X_train_raw, y_train, fold[train_mask], C_GRID)
        best_Cs.append(best_C)
 
        # 4. SVM final
        clf = SVC(kernel="rbf", C=best_C, gamma="scale", random_state=42)
        clf.fit(X_train, y_train)
 
        acc = accuracy_score(y_test, clf.predict(X_test))
        fold_scores.append(acc)
        print(f"  Fold {f}: {acc*100:.2f}%   (C = {best_C})")
 
    mean_acc = np.mean(fold_scores)
    std_acc  = np.std(fold_scores)
 
    all_results[relation] = {
        "fold_scores"   : fold_scores,
        "mean_accuracy" : mean_acc,
        "std_accuracy"  : std_acc,
        "best_Cs"       : best_Cs,
    }
    print(f"  Mean : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
 
# ─── Récapitulatif ────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  RÉCAPITULATIF FINAL")
print(f"{'='*55}")
print(f"{'Relation':<20} {'Mean Acc':>10} {'Std':>8}")
print("-" * 42)
for relation, res in all_results.items():
    print(f"{relation:<20} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")
 
overall_acc = np.mean([res["mean_accuracy"] for res in all_results.values()])
 
print(f"\n{'='*55}")
print("  ACCURACY GLOBALE — HIST-LBP + SVM v3")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  Hist-LBP + SVM v3 Accuracy : {overall_acc*100:.2f}%")
 
# ─── Diagnostic : vérifier la forme des features ─────────────────────────────
print(f"\n  [Diagnostic] Feature vector shape : {ux.shape}")
print(f"  [Diagnostic] Feature value range  : [{ux.min():.4f}, {ux.max():.4f}]")
print(f"  [Diagnostic] Expected range       : [0.0, 1.0] si L1-normalisé")