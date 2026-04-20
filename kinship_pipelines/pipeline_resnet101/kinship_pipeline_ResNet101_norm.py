import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from tensorflow.keras import layers, models
import random
import tensorflow as tf

# Reproductibilité
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ─── Chemins ────────────────────────────────────────────────────────────────
PKL_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage profond\ResNet101\resnet101_embeddings"
MAT_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "FD": {"pkl": f"{PKL_DIR}\\ResNet101_FD.pkl", "mat": f"{MAT_DIR}\\LBP_fd.mat"},
    "FS": {"pkl": f"{PKL_DIR}\\ResNet101_FS.pkl", "mat": f"{MAT_DIR}\\LBP_fs.mat"},
    "MD": {"pkl": f"{PKL_DIR}\\ResNet101_MD.pkl", "mat": f"{MAT_DIR}\\LBP_md.mat"},
    "MS": {"pkl": f"{PKL_DIR}\\ResNet101_MS.pkl", "mat": f"{MAT_DIR}\\LBP_ms.mat"},
}

# ─── CNN ────────────────────────────────────────────────────────────────────
def build_cnn1d(input_dim):
    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu', input_shape=(input_dim, 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ─── Normalisations ──────────────────────────────────────────────────────────
def decimal_scaling_fit(X):
    max_abs = np.max(np.abs(X), axis=0)
    j = np.ceil(np.log10(max_abs + 1e-12))
    j[np.isinf(j)] = 0
    return j

def decimal_scaling_transform(X, j):
    return X / (10 ** j)

def logistic_transform(X):
    X = np.clip(X, -500, 500)
    return 1 / (1 + np.exp(-X))

def sqrt_transform(X):
    return np.sqrt(X)

# ─── Méthodes ───────────────────────────────────────────────────────────────
methods = ["none", "zscore", "minmax", "decimal", "logistic", "power", "sqrt"]

final_results = {}

# ─── Boucle principale ───────────────────────────────────────────────────────
for method in methods:
    print(f"\n{'#'*60}")
    print(f"   MÉTHODE : {method.upper()}")
    print(f"{'#'*60}")

    relation_results = {}

    for relation, paths in RELATIONS.items():
        print(f"\n--- Relation : {relation} ---")

        with open(paths["pkl"], "rb") as f:
            emb_dict = pickle.load(f)

        sorted_keys = sorted(emb_dict.keys())
        ux = np.array([emb_dict[k] for k in sorted_keys])

        mat = sio.loadmat(paths["mat"])
        idxa = mat['idxa'].flatten() - 1
        idxb = mat['idxb'].flatten() - 1
        fold = mat['fold'].flatten()
        matches = mat['matches'].flatten()

        X = np.abs(ux[idxa] - ux[idxb])
        X = X.reshape(X.shape[0], X.shape[1], 1)
        y = matches

        fold_scores = []

        for f in range(1, 6):
            train_mask = fold != f
            test_mask = fold == f

            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

            # ─── Normalisation ─────────────────────────
            X_train_2D = X_train.reshape(X_train.shape[0], -1)
            X_test_2D  = X_test.reshape(X_test.shape[0], -1)

            if method == "zscore":
                scaler = StandardScaler()
                X_train_2D = scaler.fit_transform(X_train_2D)
                X_test_2D  = scaler.transform(X_test_2D)

            elif method == "minmax":
                scaler = MinMaxScaler()
                X_train_2D = scaler.fit_transform(X_train_2D)
                X_test_2D  = scaler.transform(X_test_2D)

            elif method == "decimal":
                j = decimal_scaling_fit(X_train_2D)
                X_train_2D = decimal_scaling_transform(X_train_2D, j)
                X_test_2D  = decimal_scaling_transform(X_test_2D, j)

            elif method == "logistic":
                X_train_2D = logistic_transform(X_train_2D)
                X_test_2D  = logistic_transform(X_test_2D)

            elif method == "power":
                scaler = PowerTransformer(method='yeo-johnson')
                X_train_2D = scaler.fit_transform(X_train_2D)
                X_test_2D  = scaler.transform(X_test_2D)

            elif method == "sqrt":
                X_train_2D = sqrt_transform(X_train_2D)
                X_test_2D  = sqrt_transform(X_test_2D)

            # none = pas de normalisation

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

        mean_acc = np.mean(fold_scores)
        relation_results[relation] = mean_acc

        print(f"  Mean Accuracy {relation}: {mean_acc * 100:.2f}%")

    # ─── Accuracy globale ─────────────────────────────
    overall = np.mean(list(relation_results.values()))
    final_results[method] = {
        "relations": relation_results,
        "overall": overall
    }

    print(f"\n>>> Accuracy globale ({method}) : {overall*100:.2f}%")

# ─── TABLEAU FINAL ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("   TABLEAU FINAL")
print(f"{'='*60}")

for method, res in final_results.items():
    print(f"\nMéthode : {method}")
    for rel, acc in res["relations"].items():
        print(f"  {rel} : {acc*100:.2f}%")
    print(f"  Global : {res['overall']*100:.2f}%")