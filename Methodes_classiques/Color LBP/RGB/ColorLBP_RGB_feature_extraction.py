import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle


# Fonction pour comparer le pixel central avec ses voisins
def get_pixel(img, center, x, y):
    return 1 if img[x][y] >= center else 0



# Calcul LBP uniforme (59 bins)
def lbp_pixel(img, x, y):

    center = img[x][y]

    val_ar = [
        get_pixel(img, center, x-1, y-1),
        get_pixel(img, center, x-1, y),
        get_pixel(img, center, x-1, y+1),
        get_pixel(img, center, x, y+1),
        get_pixel(img, center, x+1, y+1),
        get_pixel(img, center, x+1, y),
        get_pixel(img, center, x+1, y-1),
        get_pixel(img, center, x, y-1)
    ]

    transitions = sum(val_ar[i] != val_ar[(i + 1) % 8] for i in range(8))

    if transitions <= 2:
        value = 0
        for i in range(8):
            value += val_ar[i] * (1 << i)
        return value
    else:
        return 256  


# Extraction Color LBP RGB
def extract_color_lbp_rgb(image_path, show=False):

    # Lire l'image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    # Séparation des canaux
    B, G, R = cv2.split(img_bgr)
    channels = [R, G, B]
    histograms = []

    # Indices des motifs uniformes
    uniform_indices = []
    for i in range(256):
        binary = [(i >> j) & 1 for j in range(8)]
        transitions = sum(binary[j] != binary[(j + 1) % 8] for j in range(8))
        if transitions <= 2:
            uniform_indices.append(i)

    # Calcul LBP uniforme pour chaque canal
    for ch in channels:

        h, w = ch.shape
        img_lbp = np.zeros((h, w), dtype=np.uint16)

        # Ignorer les bordures
        for i in range(1, h-1):
            for j in range(1, w-1):
                img_lbp[i, j] = lbp_pixel(ch, i, j)

        valid = img_lbp[1:h-1, 1:w-1].ravel()
        hist, _ = np.histogram(valid, bins=257, range=(0, 257))

        feature = np.zeros(len(uniform_indices) + 1)

        for idx, pattern in enumerate(uniform_indices):
            feature[idx] = hist[pattern]

        feature[-1] = hist[256]

        histograms.append(feature)

    # Concaténation des 3 vecteurs
    color_lbp_vector = np.concatenate(histograms)


    # Affichage images + histogrammes
    if show:

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Ligne 1 : Images
        axes[0,0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title("Image originale")
        axes[0,0].axis("off")

        axes[0,1].imshow(R, cmap='gray')
        axes[0,1].set_title("Canal R")
        axes[0,1].axis("off")

        axes[0,2].imshow(G, cmap='gray')
        axes[0,2].set_title("Canal G")
        axes[0,2].axis("off")

        axes[0,3].imshow(B, cmap='gray')
        axes[0,3].set_title("Canal B")
        axes[0,3].axis("off")

        # Ligne 2 : Histogrammes
        axes[1,0].bar(range(len(histograms[0])), histograms[0])
        axes[1,0].set_title("Histogramme R")

        axes[1,1].bar(range(len(histograms[1])), histograms[1])
        axes[1,1].set_title("Histogramme G")

        axes[1,2].bar(range(len(histograms[2])), histograms[2])
        axes[1,2].set_title("Histogramme B")

        axes[1,3].bar(range(len(color_lbp_vector)), color_lbp_vector)
        axes[1,3].set_title("Histogramme combiné RGB")

        plt.tight_layout()
        plt.show()

    return color_lbp_vector



# Programme principal
if __name__ == "__main__":

    dataset_path = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
    output_path = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Color LBP\RGB\Color_LBP_RGB_feature_vectors"

    relations = ["MS", "MD", "FS", "FD"]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_images = []

    # Parcourir toute la base de données
    for relation in relations:

        relation_path = os.path.join(dataset_path, relation)
        vectors = []

        for file in os.listdir(relation_path):

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(relation_path, file)
            vec = extract_color_lbp_rgb(image_path)

            if vec is not None:
                vectors.append(vec)
                all_images.append((relation, file, image_path, vec))

        vectors = np.array(vectors)

        save_path = os.path.join(output_path, f"ColorLBP_RGB_{relation}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(vectors, f)

        print(f"{relation} saved → {vectors.shape}")

    # Exemple aléatoire
    random_sample = random.choice(all_images)
    relation, filename, path, vector = random_sample

    print("Relation:", relation)
    print("Filename:", filename)
    print("Vector length:", len(vector))
    print("Sum:", np.sum(vector))
    print(vector)

    extract_color_lbp_rgb(path, show=True)