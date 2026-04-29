import os
import pickle
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ① PATHS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE"
OUTPUT_DIR = os.path.join(BASE_DIR, "Methodes_classiques", "Hist-LDZP", "HistZigZag")
MAT_DIR    = os.path.join(BASE_DIR, "lbp")

CFG_TAG    = "ps32_ss4" 
RELATIONS  = ["FD", "FS", "MD", "MS"]

BATCH_SIZE = 32
LEARNING_RATE = 5e-4
EPOCHS = 200
PATIENCE = 25

random.seed(42); np.random.seed(42); torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# ② OPTIMIZED NORMALIZATION & MULTI-METRIC PERFUSION
# ══════════════════════════════════════════════════════════════════════════════

def apply_dual_norm(X):
    """
    Z-Score + MinMax Normalization: Zero-centers and scales ZZ-LBP features.
    Better for LCNN convergence than simple L2-Hys.
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-8
    X = (X - mean) / std
    
    x_min = np.min(X, axis=1, keepdims=True)
    x_max = np.max(X, axis=1, keepdims=True)
    X = (X - x_min) / (x_max - x_min + 1e-8)
    return X

def perfusion_fusion(feats_p, feats_c):
    """
    Four-Stream Perfusion: Absolute Difference, Product, Squared Difference, and Chi-Square.
    This captures both linear and non-linear hereditary relationships.
    """
    abs_diff = np.abs(feats_p - feats_c)
    product  = feats_p * feats_c
    sq_diff  = (feats_p - feats_c) ** 2
    
    # Chi-Square: weighted distribution difference
    chi_comp = (sq_diff) / (feats_p + feats_c + 1e-8)
    
    return np.concatenate([abs_diff, product, sq_diff, chi_comp], axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# ③ BOOSTED LCNN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d, max(1, d // r)), nn.ReLU(),
            nn.Linear(max(1, d // r), d), nn.Sigmoid()
        )
    def forward(self, x): return x * self.se(x)

class KinshipLCNN(nn.Module):
    def __init__(self, input_dim, local_out=256):
        super().__init__()
        # Swapped ReLU for LeakyReLU to prevent dead neurons in large fused vectors
        self.branch = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.1), nn.Dropout(0.35),
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.LeakyReLU(0.1), SEBlock(512), nn.Dropout(0.25),
            nn.Linear(512, local_out),  nn.BatchNorm1d(local_out), nn.LeakyReLU(0.1), SEBlock(local_out), nn.Dropout(0.20),
        )
        self.res = nn.Sequential(nn.Linear(local_out, 128), nn.BatchNorm1d(128))
        self.head = nn.Sequential(
            nn.Linear(local_out, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.20),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.15),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        h = self.branch(x)
        r = self.res(h)
        # Standard head flow
        for i in range(6):
            h = self.head[i](h)
        h = self.head[6](h) + r 
        h = self.head[7](h); h = self.head[8](h)
        return h.squeeze(1)

# ══════════════════════════════════════════════════════════════════════════════
# ④ TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def train_one_fold(X_train, y_train, X_val, y_val, input_dim):
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_va = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_va = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE)

    model = KinshipLCNN(input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0
    wait = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = torch.sigmoid(model(xb))
                preds.extend((out > 0.5).cpu().numpy())
                targets.extend(yb.cpu().numpy())
        
        acc = accuracy_score(targets, preds)
        if acc > best_acc:
            best_acc = acc
            wait = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE: break
            
    model.load_state_dict(best_model_state)
    return model

# ══════════════════════════════════════════════════════════════════════════════
# ⑤ MAIN EXECUTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results = {}

    for rel in RELATIONS:
        print(f"\n--- Processing Relation: {rel} ---")
        
        pkl_path = os.path.join(OUTPUT_DIR, CFG_TAG, f"HistZigZag_{rel}.pkl")
        if not os.path.exists(pkl_path):
            print(f"Error: {pkl_path} not found.")
            continue
            
        with open(pkl_path, "rb") as f:
            feats = np.array(pickle.load(f), dtype=np.float32)
            
        mat_path = os.path.join(MAT_DIR, f"LBP_{rel.lower()}.mat")
        if not os.path.exists(mat_path):
            print(f"Error: {mat_path} not found.")
            continue
            
        mat = sio.loadmat(mat_path)
        idxa = mat['idxa'].flatten() - 1
        idxb = mat['idxb'].flatten() - 1
        folds = mat['fold'].flatten()
        y = mat['matches'].flatten()
        
        # New Dual Norm Logic
        P, C = feats[idxa], feats[idxb]
        P_norm = apply_dual_norm(P)
        C_norm = apply_dual_norm(C)
        
        # New 4-Metric Perfusion
        X = perfusion_fusion(P_norm, C_norm)
        
        fold_accs = []
        for f in range(1, 6):
            tr_mask, te_mask = (folds != f), (folds == f)
            val_fold_idx = f % 5 + 1
            inner_val_mask = (folds[tr_mask] == val_fold_idx)
            
            X_train_full, y_train_full = X[tr_mask], y[tr_mask]
            
            model = train_one_fold(
                X_train_full[~inner_val_mask], y_train_full[~inner_val_mask], 
                X_train_full[inner_val_mask],  y_train_full[inner_val_mask],  
                input_dim=X.shape[1]
            )
            
            model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(X[te_mask], dtype=torch.float32).to(DEVICE)
                test_out = torch.sigmoid(model(test_tensor))
                fold_acc = accuracy_score(y[te_mask], (test_out > 0.5).cpu().numpy())
                fold_accs.append(fold_acc)
                print(f"  Fold {f} Accuracy: {fold_acc*100:.2f}%")
                
        mean_acc = np.mean(fold_accs)
        all_results[rel] = mean_acc
        print(f"Average {rel} Accuracy: {mean_acc*100:.2f}%")

    if all_results:
        print("\n" + "═"*40)
        print("FINAL KINSHIP RESULTS SUMMARY (Dual Norm + 4-Metric)")
        print("═"*40)
        for r, acc in all_results.items():
            print(f"{r}: {acc*100:.2f}%")
        print("-" * 40)
        print(f"OVERALL SYSTEM ACCURACY: {np.mean(list(all_results.values()))*100:.2f}%")
        print("═"*40)