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
PKL_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\VGG19\vgg19_embeddings"
MAT_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "pkl": f"{PKL_DIR}\\VGG19_FD.pkl",
        "mat": f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "pkl": f"{PKL_DIR}\\VGG19_FS.pkl",
        "mat": f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "pkl": f"{PKL_DIR}\\VGG19_MD.pkl",
        "mat": f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "pkl": f"{PKL_DIR}\\VGG19_MS.pkl",
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

# ─── Boucle principale sur les 4 relations ───────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*50}")
    print(f"  Relation : {relation}")
    print(f"{'='*50}")

    # Chargement du dict {nom_fichier: embedding}
    with open(paths["pkl"], "rb") as f:
        emb_dict = pickle.load(f)

    # Convertir en array numpy trié par nom de fichier
    sorted_keys = sorted(emb_dict.keys())
    ux = np.array([emb_dict[k] for k in sorted_keys])

    # Chargement du fichier .mat (mêmes paires/folds que LBP)
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    matches = mat['matches'].flatten()

    # Construction des paires
    X = np.abs(ux[idxa] - ux[idxb])
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = matches

    # 5-Fold Cross-Validation
    fold_scores = []
    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        model = build_cnn1d(X.shape[1])
        model.fit(X[train_mask], y[train_mask], epochs=30, batch_size=16, verbose=0)

        preds = (model.predict(X[test_mask], verbose=0) >= 0.5).astype(int).flatten()
        acc = accuracy_score(y[test_mask], preds)
        fold_scores.append(acc)
        print(f"  Fold {f}: Accuracy = {acc * 100:.2f}%")

    mean_acc = np.mean(fold_scores)
    all_results[relation] = {
        "fold_scores": fold_scores,
        "mean_accuracy": mean_acc
    }
    print(f"  Mean Accuracy : {mean_acc * 100:.2f}%")

# ─── Récapitulatif final ──────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("  RÉCAPITULATIF FINAL")
print(f"{'='*50}")
print(f"{'Relation':<20} {'Mean Accuracy':>15}")
print("-" * 36)
for relation, res in all_results.items():
    print(f"{relation:<20} {res['mean_accuracy'] * 100:>14.2f}%")

# ─── Accuracy globale VGG19 (moyenne des 4 relations) ────────────────────────
overall_acc = np.mean([res['mean_accuracy'] for res in all_results.values()])

print(f"\n{'='*50}")
print(f"  ACCURACY GLOBALE — VGG19")
print(f"{'='*50}")
print(f"  (Mean Accuracy FD + FS + MD + MS) / 4")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────")
print(f"  VGG19 Accuracy : {overall_acc * 100:.2f}%")