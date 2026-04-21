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


def extract_lbp_uniform(image_path, show=False):
    # Lire l'image 
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Conversion en niveaux de gris 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    img_lbp = np.zeros((h, w), dtype=np.uint16)

    # Ignorer les bordures pour eviter les pixels qui n'ont pas 8 voisins 
    for i in range(1, h-1):
        for j in range(1, w-1):
            img_lbp[i, j] = lbp_pixel(gray, i, j)

    valid = img_lbp[1:h-1, 1:w-1].ravel()
    histogram, _ = np.histogram(valid, bins=257, range=(0, 257))

    uniform_indices = []
    for i in range(256):
        binary = [(i >> j) & 1 for j in range(8)]
        transitions = sum(binary[j] != binary[(j + 1) % 8] for j in range(8))
        if transitions <= 2:
            uniform_indices.append(i)

    feature = np.zeros(len(uniform_indices) + 1)

    for idx, pattern in enumerate(uniform_indices):
        feature[idx] = histogram[pattern]

    feature[-1] = histogram[256]

    feature = feature.astype("float")
    #feature /= np.sum(feature)  # Normalisation

    if show:
        plt.figure(figsize=(14,4))

        plt.subplot(1,3,1)
        plt.title("Image original")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.title("Image en niveaux de gris")
        plt.imshow(gray, cmap="gray")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.title("Image LBP")
        plt.imshow(img_lbp, cmap="gray")
        plt.axis("off")
        
        plt.figure()
        plt.plot()
        plt.title("Histogram LBP")
        plt.bar(range(len(feature)), feature)

        plt.tight_layout()
        plt.show()

    return feature

# Programme principal
if __name__ == "__main__":

    dataset_path = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
    output_path = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\LBP Gray\LBP_gray_feature_vectors"
    "_vectors"

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
            vec = extract_lbp_uniform(image_path)

            if vec is not None:
                vectors.append(vec)
                all_images.append((relation, file, image_path, vec))

        vectors = np.array(vectors)

        save_path = os.path.join(output_path, f"LBP_{relation}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(vectors, f)

        print(f"{relation} saved → {vectors.shape}")
        

    # Affichage du vecteur LBP 
    random_sample = random.choice(all_images)
    relation, filename, path, vector = random_sample
     
    print("Relation:", relation)
    print("Filename:", filename)
    print("Vector length:", len(vector))
    print("Sum (should be 1):", np.sum(vector))
    print(vector)
    

    # Afficher les images 
    extract_lbp_uniform(path, show=True)
