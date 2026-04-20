#It has 3 conv layers with 64 → 128 → 256 filters so it can learn more complex patterns from the ArcFace embeddings.




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

# ─── Improved CNN 1D ─────────────────────────────────────────────────────────
def build_cnn1d(input_dim):
    model = models.Sequential([

        # Block 1 — more filters, batch norm to stabilize training
        layers.Conv1D(64, kernel_size=3, padding='same',
                      activation='relu', input_shape=(input_dim, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Block 2 — double the filters
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Block 3 — deeper representation
        layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),

        # Classifier head
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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

    # Chargement embeddings
    with open(paths["pkl"], "rb") as f:
        emb_dict = pickle.load(f)

    sorted_keys = sorted(emb_dict.keys())
    ux = np.array([emb_dict[k] for k in sorted_keys])

    # Chargement métadonnées
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    matches = mat['matches'].flatten()

    # Paires par différence absolue
    X = np.abs(ux[idxa] - ux[idxb])
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = matches

    fold_scores = []

    for f in range(1, 6):
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        train_mask = fold != f
        test_mask  = fold == f

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = build_cnn1d(X.shape[1])

        # Early stopping — stop if val_loss doesn't improve for 10 epochs
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Reduce learning rate if stuck
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )

        model.fit(
            X_train, y_train,
            epochs=100,             # more epochs, early stopping handles the rest
            batch_size=16,
            validation_split=0.15,  # use 15% of train as validation
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
print("  ACCURACY GLOBALE — ArcFace + CNN1D (improved)")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  ArcFace + CNN1D Accuracy : {overall_acc * 100:.2f}%")
