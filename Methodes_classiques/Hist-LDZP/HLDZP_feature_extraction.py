import cv2
import numpy as np
import os
import pickle
from itertools import product as iproduct


# ─── Kirsch masks (8 rotations) ───────────────────────────────────────────────
def _build_kirsch_masks() -> np.ndarray:
    base = np.array([[5,  5,  5],
                     [-3, 0, -3],
                     [-3,-3, -3]], dtype=np.float32)
    masks = []
    for k in range(8):
        M   = cv2.getRotationMatrix2D((1.0, 1.0), k * 45, 1.0)
        rot = cv2.warpAffine(base, M, (3, 3),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        masks.append(rot)
    return np.array(masks, dtype=np.float32)


KIRSCH_MASKS = _build_kirsch_masks()   # (8, 3, 3)


# ─── Uniform LBP mapping (u2, P=8, 59 bins) ──────────────────────────────────
def _build_uniform_mapping(P: int = 8) -> np.ndarray:
    n_codes = 2 ** P
    table   = np.full(n_codes, -1, dtype=np.int32)
    rank    = 0
    for i in range(n_codes):
        j    = ((i << 1) | (i >> (P - 1))) & (n_codes - 1)
        numt = bin(i ^ j).count('1')
        if numt <= 2:
            table[i] = rank
            rank += 1
    last_bin = rank   # 58
    for i in range(n_codes):
        if table[i] == -1:
            table[i] = last_bin
    return table   # (256,), values in [0, 58]


UNIFORM_MAP = _build_uniform_mapping(P=8)
N_BINS_LZP  = int(UNIFORM_MAP.max()) + 1    # 59
N_BINS_MAG  = 256                            # magnitude histogram bins

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE     = 128
SPM_LEVELS   = [1, 2, 4]        # 21 cells
N_DIRECTIONS = 8
USE_GRAY     = True              # R, G, B, Gray = 4 channels
POWER_ALPHA  = 0.5
MAG_CLIP_PCT = 95                # clip magnitude at 95th percentile before hist

_N_CH      = 4 if USE_GRAY else 3
_SPM_CELLS = sum(g * g for g in SPM_LEVELS)   # 21

# Feature breakdown:
#   LZP part  : 8 dir × 59 bins × 21 cells × 4 ch = 39,648
#   Mag part  :         256 bins × 21 cells × 4 ch = 21,504
FEATURE_DIM = (N_DIRECTIONS * N_BINS_LZP + N_BINS_MAG) * _SPM_CELLS * _N_CH
#           = (8×59 + 256) × 21 × 4 = 728 × 21 × 4 = 61,152


# =========================================================================
# Preprocessing
# =========================================================================

def preprocess_face(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize + CLAHE on L channel. Returns float32 RGB [0, 255]."""
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


# =========================================================================
# LZP map for one Kirsch response
# =========================================================================

def compute_lzp_map(response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply ZigZag LBP thresholding to one Kirsch response map.

    Neighbor bit order (ZigZag): NW NE E SE S SW W N
    Returns
    -------
    lzp_map    : uint8   (H, W)  uniform LZP code in [0, 58]
    weight_map : float32 (H, W)  absolute Kirsch response magnitude
    """
    h, w = response.shape
    c    = response[1:-1, 1:-1]

    code  = np.zeros((h - 2, w - 2), dtype=np.int32)
    code += ((response[0:-2, 0:-2] > c).astype(np.int32)) << 0   # NW
    code += ((response[0:-2, 2:  ] > c).astype(np.int32)) << 1   # NE
    code += ((response[1:-1, 2:  ] > c).astype(np.int32)) << 2   # E
    code += ((response[2:,   2:  ] > c).astype(np.int32)) << 3   # SE
    code += ((response[2:,   1:-1] > c).astype(np.int32)) << 4   # S
    code += ((response[2:,   0:-2] > c).astype(np.int32)) << 5   # SW
    code += ((response[1:-1, 0:-2] > c).astype(np.int32)) << 6   # W
    code += ((response[0:-2, 1:-1] > c).astype(np.int32)) << 7   # N

    full_code = np.zeros((h, w), dtype=np.int32)
    full_code[1:-1, 1:-1] = code

    lzp_map    = UNIFORM_MAP[full_code].astype(np.uint8)
    weight_map = np.abs(response).astype(np.float32)
    return lzp_map, weight_map


# =========================================================================
# Per-channel feature: LZP-SPM + Magnitude-SPM
# =========================================================================

def extract_channel_feature(channel: np.ndarray,
                             spm_levels: list = SPM_LEVELS) -> np.ndarray:
    """
    For one image channel, compute:
      A) Per-direction SPM histogram of uniform LZP codes (magnitude-weighted)
         Shape: (8 × 21 × 59,) = (9,912,)
      B) SPM histogram of overall Kirsch magnitude (direction-collapsed)
         Shape: (21 × 256,) = (5,376,)
    Concatenate A + B → (15,288,) per channel.
    All cells are L2-normalized individually.

    Returns float64 array of shape (15,288,)
    """
    ch     = channel.astype(np.float32)
    h, w   = ch.shape

    # ── Accumulate all 8 direction responses for magnitude histogram ─────
    # We build the combined magnitude map across all directions (max response)
    all_responses = []
    lzp_feats     = []

    for mask in KIRSCH_MASKS:
        resp         = cv2.filter2D(ch, -1, mask, borderType=cv2.BORDER_REFLECT)
        lzp, wgt     = compute_lzp_map(resp)
        all_responses.append(wgt)

        # ── A: direction-wise SPM of uniform LZP ─────────────────────────
        dir_parts = []
        for g in spm_levels:
            cell_h = h // g
            cell_w = w // g
            for row, col in iproduct(range(g), range(g)):
                r0, r1 = row * cell_h, (row + 1) * cell_h
                c0, c1 = col * cell_w, (col + 1) * cell_w
                codes  = lzp[r0:r1, c0:c1].ravel().astype(np.int32)
                wgts   = wgt[r0:r1, c0:c1].ravel().astype(np.float64)
                hist   = np.bincount(codes, weights=wgts,
                                     minlength=N_BINS_LZP)[:N_BINS_LZP].astype(np.float64)
                norm   = np.linalg.norm(hist)
                if norm > 0:
                    hist /= norm
                dir_parts.append(hist)
        lzp_feats.append(np.concatenate(dir_parts))   # (21×59,) = (1,239,)

    lzp_part = np.concatenate(lzp_feats)   # (8×1,239,) = (9,912,)

    # ── B: SPM histogram of max-pooled magnitude across all 8 directions ─
    # Max-pool gives the dominant edge strength at each pixel
    mag_map = np.stack(all_responses, axis=0).max(axis=0)   # (H, W)

    # Clip at 95th percentile to reduce outlier effect, then normalize to [0,255]
    clip_val = np.percentile(mag_map, MAG_CLIP_PCT)
    if clip_val > 0:
        mag_map = np.clip(mag_map, 0, clip_val)
        mag_map = (mag_map / clip_val * 255.0).astype(np.float32)

    mag_parts = []
    for g in spm_levels:
        cell_h = h // g
        cell_w = w // g
        for row, col in iproduct(range(g), range(g)):
            r0, r1 = row * cell_h, (row + 1) * cell_h
            c0, c1 = col * cell_w, (col + 1) * cell_w
            cell   = mag_map[r0:r1, c0:c1].ravel()
            hist, _= np.histogram(cell, bins=N_BINS_MAG, range=(0, 256))
            hist   = hist.astype(np.float64)
            norm   = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
            mag_parts.append(hist)
    mag_part = np.concatenate(mag_parts)   # (21×256,) = (5,376,)

    return np.concatenate([lzp_part, mag_part])   # (15,288,)


# =========================================================================
# Full pipeline for one image
# =========================================================================

def extract_histldzp_v4(image_path: str) -> np.ndarray | None:
    """
    Extract HistLDZP-v4 feature vector.

    Channels: R, G, B, Gray  (4 total)
    Per channel:
      - 8-direction uniform LZP SPM histograms (magnitude-weighted)  → 9,912
      - Max-pooled Kirsch magnitude SPM histogram                     → 5,376
      → 15,288 per channel
    Total: 15,288 × 4 = 61,152
    Power normalization: sign(x) × |x|^0.5

    Returns float32 array shape (61,152,)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img = preprocess_face(img_bgr)   # (128, 128, 3) float32 RGB

    channels = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]
    if USE_GRAY:
        gray = (0.299 * img[:, :, 0]
                + 0.587 * img[:, :, 1]
                + 0.114 * img[:, :, 2]).astype(np.float32)
        channels.append(gray)

    ch_feats = [extract_channel_feature(ch) for ch in channels]
    feature  = np.concatenate(ch_feats)                          # (61,152,)
    feature  = np.sign(feature) * (np.abs(feature) ** POWER_ALPHA)
    return feature.astype(np.float32)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":

    DATASET_PATH = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
    OUTPUT_DIR   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes_classiques\Hist-LDZP\HLDZP_feature_vectors"

    RELATIONS = ["FD", "FS", "MD", "MS"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"IMG_SIZE     : {IMG_SIZE}")
    print(f"N_DIRECTIONS : {N_DIRECTIONS}")
    print(f"N_BINS_LZP   : {N_BINS_LZP}  (uniform u2 mapping)")
    print(f"N_BINS_MAG   : {N_BINS_MAG}  (magnitude histogram)")
    print(f"SPM_LEVELS   : {SPM_LEVELS}  → {_SPM_CELLS} cells")
    print(f"CHANNELS     : {_N_CH}  (R, G, B{', Gray' if USE_GRAY else ''})")
    print(f"FEATURE_DIM  : {FEATURE_DIM:,}  per image")
    print(f"  LZP part   : {N_DIRECTIONS * N_BINS_LZP * _SPM_CELLS * _N_CH:,}")
    print(f"  Mag part   : {N_BINS_MAG * _SPM_CELLS * _N_CH:,}")
    print(f"Pair dim     : {FEATURE_DIM * 2 + 1:,}  (diff+prod+cos)")
    print(f"Pair RAM     : {(FEATURE_DIM*2+1) * 500 * 8 / 1e6:.0f} MB  ✓")
    print()

    for relation in RELATIONS:
        rel_path = os.path.join(DATASET_PATH, relation)
        vectors  = []

        files = sorted([
            f for f in os.listdir(rel_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        print(f"{'─'*55}")
        print(f"[{relation}]  {len(files)} images")

        for k, fname in enumerate(files):
            vec = extract_histldzp_v4(os.path.join(rel_path, fname))
            if vec is not None:
                vectors.append(vec)
            else:
                print(f"  ⚠ Skipped: {fname}")
            if (k + 1) % 50 == 0 or (k + 1) == len(files):
                print(f"  {k+1}/{len(files)} processed")

        vectors  = np.array(vectors, dtype=np.float32)
        pkl_path = os.path.join(OUTPUT_DIR, f"HistLDZP_{relation}.pkl")

        with open(pkl_path, "wb") as f:
            pickle.dump(vectors, f)

        print(f"  Shape   : {vectors.shape}")
        print(f"  Range   : [{vectors.min():.4f}, {vectors.max():.4f}]")
        print(f"  File sz : {os.path.getsize(pkl_path) / 1e6:.1f} MB")
        print(f"  Saved  → {pkl_path}")

    print("\nExtraction complete.")
    print(f"Feature dim : {FEATURE_DIM:,}")