import os
import pickle
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

RELATIONS = ["FD", "FS", "MD", "MS"]

model = ResNet50(weights='imagenet', include_top=False, pooling='max')

def get_embedding(img_path):
    img       = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return model.predict(img_array, verbose=0)[0]

if __name__ == "__main__":

    dataset_path = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
    output_dir   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\ResNet50\resnet50_embeddings"

    os.makedirs(output_dir, exist_ok=True)

    for relation in RELATIONS:
        rel_folder = os.path.join(dataset_path, relation)
        embeddings = {}

        files = sorted([f for f in os.listdir(rel_folder)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])

        print(f"\n{relation} — {len(files)} images")

        for k, file in enumerate(files):
            img_path = os.path.join(rel_folder, file)
            try:
                emb = get_embedding(img_path)
                embeddings[file] = emb
            except Exception as e:
                print(f"  Erreur: {file} -> {e}")

            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(files)} processed")

        output_file = os.path.join(output_dir, f"ResNet50_{relation}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(embeddings, f)

        print(f"  Saved: ResNet50_{relation}.pkl  ({len(embeddings)} images)")
        print(f"  First file: {list(embeddings.keys())[0]}")
        print(f"  Last file : {list(embeddings.keys())[-1]}")
        print(f"  Embedding dim: {list(embeddings.values())[0].shape}")

    print("\nExtraction complete.")