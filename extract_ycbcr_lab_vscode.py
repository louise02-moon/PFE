# ============================================================
#  HistZigZag-LBP EXTRACTION — ps16_ss8
#  Channels : Y, Cr, Cb + L, a, b  (6 channels)
#  Output   : /home/nadjia/HistZigZag/ycbcr_lab/
# ============================================================

import os, pickle, time
import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
OUT_DIR      = "/home/nadjia/HistZigZag/ycbcr_lab"

# ── Config ───────────────────────────────────────────────────
PATCH_SIZE = 16
STEP_SIZE  = 8
NUM_BINS   = 59
IMAGE_SIZE = (64, 64)
RELATIONS  = ["FD", "FS", "MD", "MS"]

# ── ZigZag LBP helpers ───────────────────────────────────────
def build_uniform_table():
    table = np.zeros(256, dtype=np.int32)
    uid = 0
    for code in range(256):
        bits = [(code >> j) & 1 for j in range(8)]
        t = sum(bits[i] != bits[(i+1) % 8] for i in range(8))
        if t <= 2: table[code] = uid; uid += 1
        else: table[code] = NUM_BINS - 1
    return table

ULBP_TABLE = build_uniform_table()

def zigzag_indices(n):
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n-1), max(0, s-n+1)-1, -1):
                indices.append((i, s - i))
        else:
            for i in range(max(0, s-n+1), min(s, n-1)+1):
                indices.append((i, s - i))
    return np.array(indices)

ZZ_IDX   = zigzag_indices(PATCH_SIZE)
_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]

def lbp_fast(channel):
    h, w = channel.shape
    lbp  = np.zeros((h, w), dtype=np.uint8)
    c    = channel[1:-1, 1:-1]
    for bit, (dr, dc) in enumerate(_OFFSETS):
        neighbor = channel[1+dr:h-1+dr, 1+dc:w-1+dc]
        lbp[1:-1, 1:-1] |= np.uint8((neighbor >= c).astype(np.uint8) << bit)
    return lbp

def extract_patches(lbp_map):
    h, w     = lbp_map.shape
    rows     = range(0, h - PATCH_SIZE + 1, STEP_SIZE)
    cols     = range(0, w - PATCH_SIZE + 1, STEP_SIZE)
    s_h, s_w = lbp_map.strides
    shape    = (len(rows), len(cols), PATCH_SIZE, PATCH_SIZE)
    strides  = (s_h * STEP_SIZE, s_w * STEP_SIZE, s_h, s_w)
    patches  = np.lib.stride_tricks.as_strided(lbp_map, shape=shape, strides=strides)
    return patches.reshape(len(rows) * len(cols), PATCH_SIZE, PATCH_SIZE)

def patches_to_hists(patches):
    r, c  = ZZ_IDX[:, 0], ZZ_IDX[:, 1]
    zz    = patches[:, r, c].astype(np.int32)
    n     = patches.shape[0]
    hists = np.zeros((n, NUM_BINS), dtype=np.float32)
    for i in range(n):
        np.add.at(hists[i], zz[i], 1)
    hists /= hists.sum(axis=1, keepdims=True) + 1e-8
    return hists

def extract_histzigzag(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, IMAGE_SIZE)

    # 6 channels: Y, Cr, Cb + L, a, b
    channels = [
        *cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)),       # Y, Cr, Cb
        *cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab)),         # L, a, b
    ]

    all_hists = []
    for ch in channels:
        lbp     = lbp_fast(ch.astype(np.float32))
        ulbp    = ULBP_TABLE[lbp]
        patches = extract_patches(ulbp)
        all_hists.append(patches_to_hists(patches))
    return np.concatenate(all_hists, axis=1).ravel().astype(np.float32)

# ── Run ──────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Extracting: YCrCb+LAB | patch={PATCH_SIZE} step={STEP_SIZE}")
print(f"Output: {OUT_DIR}\n")

t0 = time.time()
for rel in RELATIONS:
    rel_path = os.path.join(DATASET_PATH, rel)
    if not os.path.isdir(rel_path):
        print(f"  [ERROR] not found: {rel_path}"); continue
    files   = sorted(f for f in os.listdir(rel_path)
                     if f.lower().endswith((".jpg", ".png", ".jpeg")))
    vectors = []
    for k, fname in enumerate(files):
        vec = extract_histzigzag(os.path.join(rel_path, fname))
        if vec is not None:
            vectors.append(vec)
        if (k+1) % 100 == 0:
            print(f"  {rel} {k+1}/{len(files)}")
    vectors = np.array(vectors, dtype=np.float32)
    out = os.path.join(OUT_DIR, f"HistZigZag_{rel}.pkl")
    with open(out, "wb") as fh:
        pickle.dump(vectors, fh)
    print(f"  {rel}: {vectors.shape}  -> saved to {out}")

print(f"\nDone in {time.time()-t0:.1f}s")
