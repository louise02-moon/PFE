import numpy as np
import cv2
import os
import pickle
import random
import matplotlib.pyplot as plt
from scipy.fftpack import fft2


#Fonction d'extraction
def extract_lpq(image_path, show=False):

    img = cv2.imread(image_path)
    if img is None:
        return None
    

    # Conversion en niveaux de gris 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)
    h, w = gray.shape

    #Eviter les bordures
    padded = np.pad(gray, 1, mode='reflect')

    lpq_img = np.zeros((h, w), dtype=np.uint8)

    #Calcul des pixels 
    for i in range(h):
        for j in range(w):

            window = padded[i:i+3, j:j+3]
            fft_coeffs = fft2(window)

            phases = [
                np.angle(fft_coeffs[0, 1]),
                np.angle(fft_coeffs[1, 0]),
                np.angle(fft_coeffs[1, 1]),
                np.angle(fft_coeffs[1, 2])
            ]

            code = 0
            for idx, phase in enumerate(phases):

                if -np.pi/4 <= phase < np.pi/4:
                    q = 0
                elif np.pi/4 <= phase < 3*np.pi/4:
                    q = 1
                elif phase >= 3*np.pi/4 or phase < -3*np.pi/4:
                    q = 2
                else:
                    q = 3

                code |= (q << (2*idx))

            lpq_img[i, j] = code
            
            
    #L'histogramme 
    histogram, _ = np.histogram(lpq_img.ravel(), bins=256, range=(0, 256))

    histogram = histogram.astype("float")

    #Normalisation
    #histogram /= np.sum(histogram) 

    if show:
        plt.figure(figsize=(14,4))

        plt.subplot(1,3,1)
        plt.title("Image original")
        plt.imshow(img)
        plt.axis("off")
        
        plt.subplot(1,3,2)
        plt.title("Image en niveaux de gris")
        plt.imshow(gray, cmap ="gray")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.title("Image LPQ")
        plt.imshow(lpq_img, cmap="gray")
        plt.axis("off")
        
        plt.figure()
        plt.plot()
        plt.title("Histogram")
        plt.bar(range(len(histogram)), histogram)

        plt.tight_layout()
        plt.show()

    return histogram

#Programme Principal
if __name__ == "__main__":

    dataset_path = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
    output_path = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\LPQ\LPQ_feature_vectors"

    relations = ["MS", "MD", "FS", "FD"]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_images = []

    for relation in relations:

        relation_path = os.path.join(dataset_path, relation)
        vectors = []

        for file in os.listdir(relation_path):

            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(relation_path, file)
            vec = extract_lpq(image_path)

            if vec is not None:
                vectors.append(vec)
                all_images.append((relation, file, image_path, vec))

        vectors = np.array(vectors)

        save_path = os.path.join(output_path, f"LPQ_{relation}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(vectors, f)
        print(f"{relation} saved → {vectors.shape}")

    #Example
    random_sample = random.choice(all_images)
    relation, filename, path, vector = random_sample

    print("Relation:", relation)
    print("Filename:", filename)
    print("Vector length:", len(vector))
    print(vector)

    extract_lpq(path, show=True)