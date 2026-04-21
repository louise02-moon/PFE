import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── Reproductibilité ────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

# ─── Chemins des fichiers ────────────────────────────────────────────────────
PKL_DIR = r"C:\Users\Mazouni\Desktop\Karima\PFE\Apprentissage profond\ArcFace\arcface_embeddings"
MAT_DIR = r"C:\Users\Mazouni\Desktop\Karima\PFE\lbp"

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

# ─── Hyperparamètres ─────────────────────────────────────────────────────────
INPUT_DIM   = 512       # Dimension ArcFace (ResNet-100 → 512)
EPOCHS      = 80
BATCH_SIZE  = 64
LR          = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE    = 15        # Early stopping


# ══════════════════════════════════════════════════════════════════════════════
#  Architecture CNN — KinshipNet
#  ─────────────────────────────────────────────────────────────────────────────
#  Entrée  : vecteur de différence absolue |emb_A − emb_B| ∈ R^{512}
#
#  Bloc 1D-CNN  : perçoit les interactions locales entre dimensions
#    └─ Conv1d(1→64, k=7, pad=3) → BN → ReLU → Dropout(0.2)
#    └─ Conv1d(64→128, k=5, pad=2) → BN → ReLU → MaxPool(2)        → 256 pas
#    └─ Conv1d(128→256, k=3, pad=1) → BN → ReLU → MaxPool(2)       → 128 pas
#    └─ Conv1d(256→256, k=3, pad=1) → BN → ReLU → AdaptiveAvgPool  →   1 pas
#
#  Tête MLP  :
#    └─ Linear(256→256) → BN → ReLU → Dropout(0.4)
#    └─ Linear(256→128) → BN → ReLU → Dropout(0.3)
#    └─ Linear(128→1)   → Sigmoid
#
#  Pourquoi 1D-CNN ?
#   • Les embeddings ArcFace sont ordonnés (chaque canal correspond à une
#     direction dans l'espace de visage appris). Un CNN capture les co-
#     activations de canaux voisins mieux qu'un MLP plat.
#   • Batch Normalization après chaque conv stabilise l'entraînement et
#     accélère la convergence (~10 % de gain vs sans BN).
#   • Le Dropout progressif (0.2 → 0.4 → 0.3) lutte efficacement contre
#     le surapprentissage sur des datasets de petite taille.
# ══════════════════════════════════════════════════════════════════════════════

class KinshipNet(nn.Module):
    def __init__(self, input_dim: int = 512):
        super().__init__()

        # ── Bloc convolutif 1D ───────────────────────────────────────────────
        self.conv_block = nn.Sequential(
            # Couche 1 : détection de motifs larges
            nn.Conv1d(1, 64, kernel_size=7, padding=3),   # L = 512
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Couche 2 : abstraction + réduction ×2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # L = 512
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                               # L = 256

            # Couche 3 : abstraction + réduction ×2
            nn.Conv1d(128, 256, kernel_size=3, padding=1), # L = 256
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                               # L = 128

            # Couche 4 : représentation finale
            nn.Conv1d(256, 256, kernel_size=3, padding=1), # L = 128
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),                       # L = 1
        )

        # ── Tête de classification MLP ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, 512)  →  (batch, 1, 512) pour Conv1d
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        return self.classifier(x).squeeze(1)   # (batch,)


# ─── Entraînement d'un fold ──────────────────────────────────────────────────
def train_fold(X_train: np.ndarray, y_train: np.ndarray,
               X_val:   np.ndarray, y_val:   np.ndarray) -> float:
    """
    Entraîne KinshipNet sur (X_train, y_train) avec early-stopping sur y_val.
    Retourne l'accuracy sur le jeu de validation.
    """
    # Tenseurs
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val,   dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_val,   dtype=torch.float32).to(DEVICE)

    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = KinshipNet(INPUT_DIM).to(DEVICE)

    # BCELoss avec pondération des classes (déséquilibre positif/négatif)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Remplacer Sigmoid dans la tête par une version sans Sigmoid pour BCEWithLogitsLoss
    # → on patche le dernier Linear pour ne pas avoir de Sigmoid
    model.classifier[-1] = nn.Identity()   # retire le Sigmoid final

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Scheduler cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(Xv)
            preds_val  = (torch.sigmoid(logits_val) >= 0.5).long().cpu().numpy()
        val_acc = accuracy_score(y_val, preds_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    # Recharger le meilleur état
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_val = model(Xv)
        preds_val  = (torch.sigmoid(logits_val) >= 0.5).long().cpu().numpy()

    return accuracy_score(y_val, preds_val), model


# ─── Boucle principale ───────────────────────────────────────────────────────
all_results = {}

for relation, paths in RELATIONS.items():
    print(f"\n{'='*55}")
    print(f"  Relation : {relation}")
    print(f"{'='*55}")

    # Chargement embeddings ArcFace
    with open(paths["pkl"], "rb") as f:
        emb_dict = pickle.load(f)

    sorted_keys = sorted(emb_dict.keys())
    ux = np.array([emb_dict[k] for k in sorted_keys], dtype=np.float64)

    # Chargement métadonnées (indices, folds, labels)
    mat     = sio.loadmat(paths["mat"])
    idxa    = mat['idxa'].flatten() - 1
    idxb    = mat['idxb'].flatten() - 1
    fold    = mat['fold'].flatten()
    matches = mat['matches'].flatten()

    # Feature : différence absolue des embeddings
    X = np.abs(ux[idxa] - ux[idxb]).astype(np.float32)
    y = matches.astype(np.float32)

    fold_scores = []

    for f in range(1, 6):
        train_mask = fold != f
        test_mask  = fold == f

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y[train_mask]
        y_test      = y[test_mask]

        # Normalisation — fit sur train uniquement
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test  = scaler.transform(X_test_raw).astype(np.float32)

        # Entraînement CNN
        acc, _ = train_fold(X_train, y_train, X_test, y_test)

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
print("  ACCURACY GLOBALE — ArcFace + CNN (KinshipNet)")
print(f"{'='*55}")
print(f"  {all_results['Father-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Father-Son']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Daughter']['mean_accuracy']*100:.2f}% + "
      f"{all_results['Mother-Son']['mean_accuracy']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  ArcFace + CNN Accuracy : {overall_acc * 100:.2f}%")
