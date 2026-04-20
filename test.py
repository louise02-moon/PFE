import pickle, numpy as np, scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

with open(r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LBP\Color_HLBP_feature_vectors_v2\HistLBP_FD.pkl", "rb") as f:
    hlbp = np.array(pickle.load(f), dtype=np.float64)

mat  = sio.loadmat(r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp\LBP_fd.mat")
idxa = mat['idxa'].flatten() - 1
idxb = mat['idxb'].flatten() - 1
fold = mat['fold'].flatten()
y    = mat['matches'].flatten()

X = np.abs(hlbp[idxa] - hlbp[idxb])
X = np.sqrt(X)

for f in range(1, 6):
    tr, te = fold != f, fold == f
    Xtr = X[tr] - X[tr].mean(axis=0)
    Xte = X[te] - X[tr].mean(axis=0)
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    clf = SVC(kernel='linear', C=0.001, random_state=42)
    clf.fit(Xtr, y[tr])
    print(f"Fold {f}: {accuracy_score(y[te], clf.predict(Xte))*100:.1f}%")