import cv2
import numpy as np
import os
import pickle


# ─── Uniform LBP lookup table (0-255 → bin 0-57 uniform, 58 non-uniform) ────
def build_uniform_lbp_table():
    table = np.zeros(256, dtype=np.int32)
    uniform_bin = 0
    for code in range(256):
        bits = [(code >> j) & 1 for j in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            table[code] = uniform_bin
            uniform_bin += 1
        else:
            table[code] = 58   # non-uniform bin
    return table

LBP_TABLE = build_uniform_lbp_table()


# ─── Compute LBP map for a single channel ────────────────────────────────────
def compute_lbp_map(channel):
    h, w = channel.shape
    lbp_map = np.zeros((h, w), dtype=np.int32)

    for x in range(1, h - 1):
        for y in range(1, w - 1):
            center = channel[x, y]
            neighbors = [
                channel[x-1, y-1], channel[x-1, y], channel[x-1, y+1],
                channel[x,   y+1],
                channel[x+1, y+1], channel[x+1, y], channel[x+1, y-1],
                channel[x,   y-1]
            ]
            bits = [1 if n >= center else 0 for n in neighbors]
            code = sum(bits[i] * (1 << i) for i in range(8))
            lbp_map[x, y] = LBP_TABLE[code]

    return lbp_map


# ─── Hist LBP with sliding window ────────────────────────────────────────────
def extract_histlbp(image_path, patch_size=32, step_size=2, num_bins=16):
    """
    Extracts Hist LBP features using a sliding window approach.
    Channels: R, G, B, H, S, V, Y, Cb, Cr (color histograms) + Grayscale LBP.
    Each patch contributes: 9 × num_bins + 59 = 9×16+59 = 203 dims (default).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)  # OpenCV: Y, Cr, Cb
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Separate channels
    R, G, B = img_rgb[:,:,0],   img_rgb[:,:,1],   img_rgb[:,:,2]
    H, S, V = img_hsv[:,:,0],   img_hsv[:,:,1],   img_hsv[:,:,2]
    Y, Cr, Cb = img_ycbcr[:,:,0], img_ycbcr[:,:,1], img_ycbcr[:,:,2]

    color_channels = [R, G, B, H, S, V, Y, Cb, Cr]

    # Precompute grayscale LBP map
    lbp_map = compute_lbp_map(img_gray.astype(np.float64))

    h, w = img_gray.shape
    feature_list = []

    # Sliding window
    for i in range(0, h - patch_size + 1, step_size):
        for j in range(0, w - patch_size + 1, step_size):

            patch_feat = []

            # ── 9 color channel histograms ────────────────────────────────────
            for ch in color_channels:
                patch = ch[i:i+patch_size, j:j+patch_size].ravel().astype(np.float64)
                hist, _ = np.histogram(patch, bins=num_bins, range=(0, 256))
                hist = hist.astype(np.float64)
                s = hist.sum()
                if s > 0:
                    hist /= s
                patch_feat.append(hist)

            # ── Grayscale LBP histogram (59 bins) ────────────────────────────
            lbp_patch = lbp_map[i:i+patch_size, j:j+patch_size].ravel()
            lbp_hist, _ = np.histogram(lbp_patch, bins=59, range=(0, 59))
            lbp_hist = lbp_hist.astype(np.float64)
            s = lbp_hist.sum()
            if s > 0:
                lbp_hist /= s

            patch_feat.append(lbp_hist)

            feature_list.append(np.concatenate(patch_feat))

    if len(feature_list) == 0:
        return None

    return np.concatenate(feature_list)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    dataset_path = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
    output_path  = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes_classiques\Hist-LBP\Color_HLBP_feature_vectors"

    relations  = ["MS", "MD", "FS", "FD"]
    patch_size = 32
    step_size  = 2
    num_bins   = 16

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for relation in relations:

        relation_path = os.path.join(dataset_path, relation)
        vectors = []

        files = [f for f in os.listdir(relation_path)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        for k, file in enumerate(files):
            image_path = os.path.join(relation_path, file)
            vec = extract_histlbp(image_path,
                                  patch_size=patch_size,
                                  step_size=step_size,
                                  num_bins=num_bins)
            if vec is not None:
                vectors.append(vec)

            if (k + 1) % 50 == 0:
                print(f"  {relation}: {k+1}/{len(files)} images processed")

        vectors = np.array(vectors)

        save_path = os.path.join(output_path, f"HistLBP_{relation}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(vectors, f)

        print(f"{relation} saved → shape: {vectors.shape}")
        print(f"  Value range: [{vectors.min():.4f}, {vectors.max():.4f}]")

    print("\nExtraction complete.")