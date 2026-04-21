import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
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


# Extraction Color LBP HSV
def extract_color_lbp_hsv(image_path, show=False):

    # Lire l'image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Conversion en espace HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)

    channels = [H, S, V]
    histograms = []

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

        # Ignorer les bordures pour eviter les pixels qui n'ont pas 8 voisins
        for i in range(1, h-1):
            for j in range(1, w-1):
                img_lbp[i, j] = lbp_pixel(ch, i, j)

        valid = img_lbp[1:h-1, 1:w-1].ravel()
        histogram, _ = np.histogram(valid, bins=257, range=(0, 257))

        feature = np.zeros(len(uniform_indices) + 1)

        for idx, pattern in enumerate(uniform_indices):
            feature[idx] = histogram[pattern]

        feature[-1] = histogram[256]

        histograms.append(feature)

    # Concatenation des 3 vecteurs
    color_feature = np.concatenate(histograms)

    if show:

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Images 
        axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title("Image originale")
        axes[0,0].axis("off")

        axes[0,1].imshow(H, cmap='gray')
        axes[0,1].set_title("Canal H")
        axes[0,1].axis("off")

        axes[0,2].imshow(S, cmap='gray')
        axes[0,2].set_title("Canal S")
        axes[0,2].axis("off")

        axes[0,3].imshow(V, cmap='gray')
        axes[0,3].set_title("Canal V")
        axes[0,3].axis("off")


        # Histogrammes
        axes[1,0].bar(range(len(histograms[0])), histograms[0])
        axes[1,0].set_title("Histogramme H")

        axes[1,1].bar(range(len(histograms[1])), histograms[1])
        axes[1,1].set_title("Histogramme S")

        axes[1,2].bar(range(len(histograms[2])), histograms[2])
        axes[1,2].set_title("Histogramme V")

        axes[1,3].bar(range(len(color_feature)), color_feature)
        axes[1,3].set_title("Histogramme combiné HSV")

        plt.tight_layout()
        plt.show()

    return color_feature


# Programme principal
if __name__ == "__main__":

    dataset_path = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
    output_path = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Color LBP\HSV\Color_LBP_HSV_feature_vectors"

    relations = ["MS", "MD", "FS", "FD"]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_images = []

    # Parcourir toute la base de donnees
    for relation in relations:

        relation_path = os.path.join(dataset_path, relation)
        vectors = []

        for file in os.listdir(relation_path):

            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(relation_path, file)
            vec = extract_color_lbp_hsv(image_path)

            if vec is not None:
                vectors.append(vec)
                all_images.append((relation, file, image_path, vec))

        vectors = np.array(vectors)

        save_path = os.path.join(output_path, f"ColorLBP_HSV_{relation}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(vectors, f)

        print(f"{relation} saved → {vectors.shape}")

    # Affichage du vecteur Color LBP HSV
    random_sample = random.choice(all_images)
    relation, filename, path, vector = random_sample

    print("Relation:", relation)
    print("Filename:", filename)
    print("Vector length:", len(vector))
    print("Sum:", np.sum(vector))
    print(vector)

    # Afficher les images
    extract_color_lbp_hsv(path, show=True)