import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models, callbacks, regularizers
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


# ─── Pair feature builder ────────────────────────────────────────────────────
def build_pair_features(ux, idxa, idxb):
    """
    ArcFace embeddings are L2-normalized unit vectors.
    Cosine similarity = dot product when vectors are unit-normalized.
    We build a rich feature vector for each pair combining:
      1. Absolute difference         |a - b|         (512 dims)
      2. Element-wise product         a * b           (512 dims)  ← captures similarity
      3. Cosine similarity score      dot(a,b)        (1 dim)     ← the key ArcFace signal
    Total: 1025 dims per pair — richer than just |a - b|
    """
    a = ux[idxa]   # (N, 512)
    b = ux[idxb]   # (N, 512)

    # L2 normalize just in case embeddings are not already normalized
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

    abs_diff   = np.abs(a_norm - b_norm)                          # (N, 512)
    elem_prod  = a_norm * b_norm                                  # (N, 512)
    cos_sim    = np.sum(a_norm * b_norm, axis=1, keepdims=True)   # (N, 1)

    return np.concatenate([abs_diff, elem_prod, cos_sim], axis=1)  # (N, 1025)


# ─── Simple CNN 1D suited for 1025-dim input ─────────────────────────────────
def build_cnn1d_cosine(input_dim):
    model = models.Sequential([

        layers.Conv1D(32, kernel_size=5, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-3),
                      input_shape=(input_dim, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-3)),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),

        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
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

    # Richer pair features using cosine similarity
    X = build_pair_features(ux, idxa, idxb)   # (N, 1025)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = matches

    print(f"  Pair feature shape: {X.shape}")

    fold_scores = []

    for f in range(1, 6):
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        train_mask = fold != f
        test_mask  = fold == f

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = build_cnn1d_cosine(X.shape[1])

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=0
        )

        model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=16,
            validation_split=0.15,
            callbacks=[early_stop, reduce_lr],
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
print("  ACCURACY GLOBALE — ArcFace + CNN1D (cosine features)")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  ArcFace + CNN1D (cosine) Accuracy : {overall_acc * 100:.2f}%")