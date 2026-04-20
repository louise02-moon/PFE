import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, callbacks
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

# ─── Pair feature builder ─────────────────────────────────────────────────────
def build_pair_features(ux, idxa, idxb):
    """
    Builds a rich 1025-dim feature vector for each pair:
      1. |a - b|    — difference         (512 dims)
      2. a * b      — shared signal      (512 dims)
      3. dot(a, b)  — cosine similarity  (  1 dim)
    """
    a = ux[idxa]
    b = ux[idxb]

    # L2 normalize
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

    abs_diff  = np.abs(a - b)
    elem_prod = a * b
    cos_sim   = np.sum(a * b, axis=1, keepdims=True)

    return np.concatenate([abs_diff, elem_prod, cos_sim], axis=1)  # (N, 1025)

# ─── EXACT original CNN architecture — nothing changed ───────────────────────
def build_cnn1d_original(input_dim):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu',
                      input_shape=(input_dim, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ─── Boucle principale ───────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*55}")
    print(f"  Relation : {relation}")
    print(f"{'='*55}")

    with open(paths["pkl"], "rb") as f:
        emb_dict = pickle.load(f)

    sorted_keys = sorted(emb_dict.keys())
    ux = np.array([emb_dict[k] for k in sorted_keys])

    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    matches = mat['matches'].flatten()

    # Richer input — same architecture, better features
    X = build_pair_features(ux, idxa, idxb)   # (N, 1025)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = matches

    print(f"  Input shape : {X.shape}")

    fold_scores = []

    for f in range(1, 6):
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        train_mask = fold != f
        test_mask  = fold == f

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = build_cnn1d_original(X.shape[1])

        # Early stopping — same patience as original training length (30 epochs)
        # gives it room to breathe without letting it overfit
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=16,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )

        preds = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()
        acc   = accuracy_score(y_test, preds)
        fold_scores.append(acc)
        print(f"  Fold {f}: {acc * 100:.2f}%")

    mean_acc = np.mean(fold_scores)
    std_acc  = np.std(fold_scores)

    all_results[relation] = {
        "fold_scores"   : fold_scores,
        "mean_accuracy" : mean_acc,
        "std_accuracy"  : std_acc,
    }
    print(f"  Mean : {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

# ─── Récapitulatif ────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  RÉCAPITULATIF FINAL")
print(f"{'='*55}")
print(f"{'Relation':<22} {'Mean Acc':>10} {'Std':>8}")
print("-" * 44)
for relation, res in all_results.items():
    print(f"{relation:<22} {res['mean_accuracy']*100:>9.2f}%"
          f" {res['std_accuracy']*100:>7.2f}%")

overall_acc = np.mean([res["mean_accuracy"] for res in all_results.values()])

print(f"\n{'='*55}")
print("  ACCURACY GLOBALE — ArcFace + Original CNN + Cosine Input")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  ArcFace + Original CNN + Cosine Input : {overall_acc * 100:.2f}%")