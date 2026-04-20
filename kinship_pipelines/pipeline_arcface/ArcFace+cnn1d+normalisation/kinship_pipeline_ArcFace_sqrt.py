import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
import random
import tensorflow as tf

# Reproductibilité
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ─── Chemins des fichiers ────────────────────────────────────────────────────
PKL_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
MAT_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "pkl": f"{PKL_DIR}\\ArcFace_FD.pkl",
        "mat": f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "pkl": f"{PKL_DIR}\\ArcFace_FS.pkl",
        "mat": f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "pkl": f"{PKL_DIR}\\ArcFace_MD.pkl",
        "mat": f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "pkl": f"{PKL_DIR}\\ArcFace_MS.pkl",
        "mat": f"{MAT_DIR}\\LBP_ms.mat",
    },
}

# ─── Modèle CNN 1D ───────────────────────────────────────────────────────────
def build_cnn1d(input_dim):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ─── Fonction transformation racine carrée ──────────────────────────────────
def sqrt_transform(X):
    return np.sqrt(X)

# ─── Boucle principale ───────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*50}")
    print(f"  Relation : {relation}")
    print(f"{'='*50}")

    with open(paths["pkl"], "rb") as f:
        emb_dict = pickle.load(f)

    sorted_keys = sorted(emb_dict.keys())
    ux = np.array([emb_dict[k] for k in sorted_keys])

    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    matches = mat['matches'].flatten()

    X = np.abs(ux[idxa] - ux[idxb])
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = matches

    fold_scores = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_train = X[train_mask]
        X_test  = X[test_mask]
        y_train = y[train_mask]
        y_test  = y[test_mask]

        # 🔥 TRANSFORMATION RACINE CARRÉE
        X_train_2D = X_train.reshape(X_train.shape[0], -1)
        X_test_2D  = X_test.reshape(X_test.shape[0], -1)

        X_train_2D = sqrt_transform(X_train_2D)
        X_test_2D  = sqrt_transform(X_test_2D)

        X_train = X_train_2D.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test  = X_test_2D.reshape(X_test.shape[0], X_test.shape[1], 1)

        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        model = build_cnn1d(X.shape[1])
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

        preds = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, preds)

        fold_scores.append(acc)
        print(f"  Fold {f}: Accuracy = {acc * 100:.2f}%")

    mean_acc = np.mean(fold_scores)
    all_results[relation] = {
        "fold_scores": fold_scores,
        "mean_accuracy": mean_acc
    }

    print(f"  Mean Accuracy : {mean_acc * 100:.2f}%")

# ─── Résumé ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("  RÉCAPITULATIF FINAL")
print(f"{'='*50}")
print(f"{'Relation':<20} {'Mean Accuracy':>15}")
print("-" * 36)

for relation, res in all_results.items():
    print(f"{relation:<20} {res['mean_accuracy'] * 100:>14.2f}%")

overall_acc = np.mean([res['mean_accuracy'] for res in all_results.values()])

print(f"\n{'='*50}")
print("  ACCURACY GLOBALE — ARCFACE (RACINE CARRÉE)")
print(f"{'='*50}")
print(f"  ArcFace Accuracy : {overall_acc * 100:.2f}%")