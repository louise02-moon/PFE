import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import random

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

# ─── Shared embedding network ─────────────────────────────────────────────────
# This sub-network processes ONE embedding.
# Both branches of the Siamese network share these exact same weights.
# Small and regularized — designed for small datasets.
def build_embedding_network(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Output: 64-dim representation of the face
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3)),
    ], name='embedding_network')
    return model


# ─── Full Siamese network ─────────────────────────────────────────────────────
def build_siamese(input_dim):
    # Two inputs — one for each person in the pair
    input_a = layers.Input(shape=(input_dim,), name='input_parent')
    input_b = layers.Input(shape=(input_dim,), name='input_child')

    # Shared network — same weights for both inputs
    shared_net = build_embedding_network(input_dim)

    # Process both embeddings through the SAME network
    encoded_a = shared_net(input_a)
    encoded_b = shared_net(input_b)

    # Compute the L1 distance between the two encoded representations
    # This is what the network learns to minimize for kin pairs
    l1_distance = layers.Lambda(
        lambda tensors: tf.abs(tensors[0] - tensors[1]),
        name='l1_distance'
    )([encoded_a, encoded_b])

    # Also add cosine similarity as extra signal
    cos_sim = layers.Dot(axes=1, normalize=True,
                         name='cosine_sim')([encoded_a, encoded_b])
    cos_sim = layers.Reshape((1,))(cos_sim)

    # Concatenate distance + cosine score
    merged = layers.Concatenate(name='merged')([l1_distance, cos_sim])

    # Final classification head
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3))(merged)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(
        inputs=[input_a, input_b],
        outputs=output,
        name='siamese_network'
    )

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

    # Chargement embeddings
    with open(paths["pkl"], "rb") as f:
        emb_dict = pickle.load(f)

    sorted_keys = sorted(emb_dict.keys())
    ux = np.array([emb_dict[k] for k in sorted_keys], dtype=np.float64)

    # L2 normalize
    ux = ux / (np.linalg.norm(ux, axis=1, keepdims=True) + 1e-8)

    # Chargement métadonnées
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    matches = mat['matches'].flatten()

    # Separate inputs — parent embeddings and child embeddings
    # The Siamese network sees both independently, not their difference
    X_a = ux[idxa]   # parent embeddings  (N, 512)
    X_b = ux[idxb]   # child  embeddings  (N, 512)
    y   = matches

    print(f"  Embedding dim : {X_a.shape[1]}")
    print(f"  Pairs         : {X_a.shape[0]}")

    fold_scores = []

    for f in range(1, 6):
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        train_mask = fold != f
        test_mask  = fold == f

        Xa_train, Xa_test = X_a[train_mask], X_a[test_mask]
        Xb_train, Xb_test = X_b[train_mask], X_b[test_mask]
        y_train,  y_test  = y[train_mask],   y[test_mask]

        model = build_siamese(X_a.shape[1])

        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=0
        )

        model.fit(
            [Xa_train, Xb_train], y_train,
            epochs=200,
            batch_size=16,
            validation_split=0.15,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        preds = (model.predict(
            [Xa_test, Xb_test], verbose=0) >= 0.5
        ).astype(int).flatten()

        acc = accuracy_score(y_test, preds)
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
print("  ACCURACY GLOBALE — ArcFace + Siamese Network")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  ArcFace + Siamese Accuracy : {overall_acc * 100:.2f}%")