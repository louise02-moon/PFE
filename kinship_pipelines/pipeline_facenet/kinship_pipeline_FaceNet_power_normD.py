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
PKL_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\FaceNet\facenet_embeddings"
MAT_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {
        "pkl": f"{PKL_DIR}\\FaceNet_FD.pkl",
        "mat": f"{MAT_DIR}\\LBP_fd.mat",
    },
    "Father-Son": {
        "pkl": f"{PKL_DIR}\\FaceNet_FS.pkl",
        "mat": f"{MAT_DIR}\\LBP_fs.mat",
    },
    "Mother-Daughter": {
        "pkl": f"{PKL_DIR}\\FaceNet_MD.pkl",
        "mat": f"{MAT_DIR}\\LBP_md.mat",
    },
    "Mother-Son": {
        "pkl": f"{PKL_DIR}\\FaceNet_MS.pkl",
        "mat": f"{MAT_DIR}\\LBP_ms.mat",
    },
}

# ─── Power normalization ──────────────────────────────────────────────────────
def power_normalization(X_train, X_test):
    # Step 1: abs — remove negatives
    X_train = np.abs(X_train)
    X_test  = np.abs(X_test)

    # Step 2: sqrt — compress large values
    X_train = np.sqrt(X_train)
    X_test  = np.sqrt(X_test)

    # Step 3: mean centering — fitted on train only
    mean_   = X_train.mean(axis=0)
    X_train = X_train - mean_
    X_test  = X_test  - mean_

    return X_train, X_test

# ─── EXACT original CNN — nothing changed ────────────────────────────────────
def build_cnn1d(input_dim):
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

    # Absolute difference — 128 dims
    X = np.abs(ux[idxa] - ux[idxb])
    y = matches

    fold_scores = []

    for f in range(1, 6):
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        train_mask = fold != f
        test_mask  = fold == f

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y[train_mask]
        y_test      = y[test_mask]

        # Power normalization — fitted on train only
        X_train, X_test = power_normalization(X_train_raw, X_test_raw)

        # Reshape for CNN
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

        # Original CNN — unchanged
        model = build_cnn1d(X_train.shape[1])

        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=16,
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
print("  ACCURACY GLOBALE — FaceNet + CNN1D + Power Norm")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  FaceNet + CNN1D + Power Norm Accuracy : {overall_acc * 100:.2f}%")