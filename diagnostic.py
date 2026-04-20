"""
HistLDZP + LCNN Diagnostic
──────────────────────────
Tests different PCA compression levels to find which gives best LCNN accuracy.
Run this before the full multi-deep fusion to know the right LDZP_PCA_DIM.
"""

import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── Paths ────────────────────────────────────────────────────────────────────
HISTLDZP_DIR = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LDZP\HLDZP_feature_vectors"
MAT_DIR      = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp"

RELATIONS = {
    "Father-Daughter": {"hldzp": f"{HISTLDZP_DIR}\\HistLDZP_FD.pkl", "mat": f"{MAT_DIR}\\LBP_fd.mat"},
    "Father-Son"     : {"hldzp": f"{HISTLDZP_DIR}\\HistLDZP_FS.pkl", "mat": f"{MAT_DIR}\\LBP_fs.mat"},
    "Mother-Daughter": {"hldzp": f"{HISTLDZP_DIR}\\HistLDZP_MD.pkl", "mat": f"{MAT_DIR}\\LBP_md.mat"},
    "Mother-Son"     : {"hldzp": f"{HISTLDZP_DIR}\\HistLDZP_MS.pkl", "mat": f"{MAT_DIR}\\LBP_ms.mat"},
}

# PCA dims to test — from very small to large
PCA_DIMS = [128, 256, 512, 1024, 2048]

# =========================================================================
# LCNN — same architecture as main pipeline (single branch for standalone)
# =========================================================================
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, max(1, dim // reduction)), nn.ReLU(),
            nn.Linear(max(1, dim // reduction), dim), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.se(x)

class LocalBranch(nn.Module):
    def __init__(self, input_dim, hidden=512, out=256, drop1=0.3, drop2=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(drop1),
            nn.Linear(hidden, out),       nn.BatchNorm1d(out),    nn.ReLU(), SEBlock(out), nn.Dropout(drop2),
        )
    def forward(self, x): return self.net(x)

class LCNN_Single(nn.Module):
    """Single-branch LCNN for HistLDZP standalone testing"""
    def __init__(self, input_dim, local_out=256):
        super().__init__()
        self.branch = LocalBranch(input_dim, hidden=512, out=local_out, drop1=0.35, drop2=0.2)
        self.fusion = nn.Sequential(
            nn.Linear(local_out, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.residual = nn.Sequential(nn.Linear(local_out, 64), nn.BatchNorm1d(64))

    def forward(self, x):
        h = self.branch(x)
        r = self.residual(h)
        h = self.fusion[0](h)
        h = self.fusion[1](h)
        h = self.fusion[2](h)
        h = self.fusion[3](h)
        h = self.fusion[4](h)
        h = self.fusion[5](h)
        h = self.fusion[6](h)
        h = h + r
        h = self.fusion[7](h)
        return self.fusion[8](h).squeeze(1)

def mixup_batch(Xb, yb, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(Xb.size(0), device=Xb.device)
    return lam * Xb + (1-lam) * Xb[idx], lam * yb + (1-lam) * yb[idx]

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        Xb, yb = mixup_batch(Xb, yb)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            p = (torch.sigmoid(model(Xb.to(DEVICE))) >= 0.5).long().cpu().numpy()
            preds.extend(p); labels.extend(yb.numpy())
    return accuracy_score(labels, preds)

def train_lcnn(X_tr, y_tr, X_va, y_va, input_dim, epochs=150, batch=32, lr=5e-4, patience=20):
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xva_t = torch.tensor(X_va, dtype=torch.float32)
    yva_t = torch.tensor(y_va, dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=batch, shuffle=False)

    model    = LCNN_Single(input_dim).to(DEVICE)
    pos_w    = torch.tensor([(y_tr==0).sum()/max((y_tr==1).sum(),1)], dtype=torch.float32).to(DEVICE)
    crit     = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt      = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)

    def lr_fn(ep):
        if ep < 10: return (ep+1)/10
        return 0.5*(1+np.cos(np.pi*(ep-10)/max(1,epochs-10)))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_fn)

    best_acc, best_state, wait = 0.0, None, 0
    for ep in range(epochs):
        train_epoch(model, tr_loader, opt, crit)
        sched.step()
        acc = evaluate(model, va_loader)
        if acc > best_acc:
            best_acc  = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model

def run_lcnn_pca(feats, y, fold, pca_dim):
    """Run 5-fold LCNN with PCA compression. PCA fitted inside each fold on train only."""
    fold_scores = []
    for f in range(1, 6):
        torch.manual_seed(42); np.random.seed(42); random.seed(42)

        train_mask = fold != f
        test_mask  = fold == f

        X_tr_raw = feats[train_mask].astype(np.float32)
        X_te_raw = feats[test_mask].astype(np.float32)
        y_train  = y[train_mask]
        y_test   = y[test_mask]

        # ── PCA compression — fitted ONLY on train fold ───────────────────
        n_comp = min(pca_dim, X_tr_raw.shape[0]-1, X_tr_raw.shape[1])
        pca    = PCA(n_components=n_comp, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_raw)
        X_te_pca = pca.transform(X_te_raw)

        # ── StandardScaler ────────────────────────────────────────────────
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_pca)
        X_te_sc = sc.transform(X_te_pca)

        # ── Build pair features after PCA ─────────────────────────────────
        # We need image-level PCA first, then build pairs
        # Reload original feats and rebuild pair features properly
        # (already done above — X_tr_sc is pair features after PCA)

        # ── Inner validation fold ─────────────────────────────────────────
        inner_fold_ids = fold[train_mask]
        inner_val_fold = f % 5 + 1
        inner_val_mask = inner_fold_ids == inner_val_fold

        X_inner_tr = X_tr_sc[~inner_val_mask]
        X_inner_va = X_tr_sc[ inner_val_mask]
        y_inner_tr = y_train[~inner_val_mask]
        y_inner_va = y_train[ inner_val_mask]

        model = train_lcnn(X_inner_tr, y_inner_tr, X_inner_va, y_inner_va,
                           input_dim=X_tr_sc.shape[1])

        te_loader = DataLoader(
            TensorDataset(torch.tensor(X_te_sc, dtype=torch.float32),
                          torch.tensor(y_test,  dtype=torch.float32)),
            batch_size=32, shuffle=False)

        acc = evaluate(model, te_loader)
        fold_scores.append(acc)

    return float(np.mean(fold_scores)), float(np.std(fold_scores))

# ─── Main ─────────────────────────────────────────────────────────────────────
print("\nBuilding pair features for all relations...")

# Pre-build pair features for all relations
relation_pairs = {}
for relation, paths in RELATIONS.items():
    with open(paths["hldzp"], "rb") as f:
        feats = np.array(pickle.load(f), dtype=np.float32)

    mat  = sio.loadmat(paths["mat"])
    idxa = mat['idxa'].flatten() - 1
    idxb = mat['idxb'].flatten() - 1
    fold = mat['fold'].flatten()
    y    = mat['matches'].flatten()

    # Absolute difference pair features — same as HistLBP pipeline
    X = np.abs(feats[idxa] - feats[idxb])
    relation_pairs[relation] = (X, y, fold)
    print(f"  {relation}: pair shape = {X.shape}")

# ─── Test each PCA dim with LCNN ─────────────────────────────────────────────
pca_results = {}

for pca_dim in PCA_DIMS:
    print(f"\n{'='*55}")
    print(f"  PCA dim = {pca_dim}")
    print(f"{'='*55}")

    relation_means = []
    for relation, (X, y, fold) in relation_pairs.items():
        mean_acc, std_acc = run_lcnn_pca(X, y, fold, pca_dim)
        relation_means.append(mean_acc)
        print(f"  {relation:<22} {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    overall = float(np.mean(relation_means))
    pca_results[pca_dim] = overall
    print(f"  {'Overall':<22} {overall*100:.2f}%")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  SUMMARY — HistLDZP + LCNN vs PCA dimension")
print(f"{'='*55}")
print(f"{'PCA Dim':<12} {'Overall Accuracy':>18}")
print("-" * 35)
for dim, acc in sorted(pca_results.items()):
    marker = " ← best" if dim == max(pca_results, key=pca_results.get) else ""
    print(f"  {dim:<10} {acc*100:>17.2f}%{marker}")

best_dim = max(pca_results, key=pca_results.get)
print(f"\n  Best PCA dim  : {best_dim}")
print(f"  Best accuracy : {pca_results[best_dim]*100:.2f}%")
print(f"  Reference     : HistLBP + SVM = 88.00%")
print(f"\n  → Use LDZP_PCA_DIM = {best_dim} in the multi-deep LCNN fusion")