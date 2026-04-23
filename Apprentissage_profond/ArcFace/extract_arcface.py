import os
import pickle
import numpy as np
from deepface import DeepFace

RELATIONS = ["FD", "FS", "MD", "MS"]

def get_embedding(img_path):
    embedding = DeepFace.represent(
        img_path=img_path,
        model_name="ArcFace",
        enforce_detection=False
    )
    return np.array(embedding[0]["embedding"])

if __name__ == "__main__":

    dataset_path = r"C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images"
    output_dir   = r"C:\Users\surface laptop 5\OneDrive\Documents\PFE\Apprentissage_profond\ArcFace\arcface_embeddings"

    os.makedirs(output_dir, exist_ok=True)

    for relation in RELATIONS:
        rel_folder = os.path.join(dataset_path, relation)

        files = sorted([f for f in os.listdir(rel_folder)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])

        print(f"\n{relation} — {len(files)} images")

        embedding_list = []   # will become a (N, 512) array — order = sorted filenames
        filenames      = []   # track which file → which row

        for k, file in enumerate(files):
            img_path = os.path.join(rel_folder, file)
            try:
                emb = get_embedding(img_path)
                embedding_list.append(emb)
                filenames.append(file)
            except Exception as e:
                print(f"  Erreur: {file} -> {e}")

            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(files)} processed")

        # ── Save as a plain float32 numpy array (version-safe) ──────────────
        # Shape: (N_images, embed_dim)  e.g. (200, 512) for ArcFace
        embeddings_array = np.array(embedding_list, dtype=np.float32)

        # .npy  → numpy-native, totally version-proof, fastest to load
        npy_path = os.path.join(output_dir, f"ArcFace_{relation}.npy")
        np.save(npy_path, embeddings_array)

        # .pkl  → also saved, but as a PLAIN ndarray (not a dict)
        #         protocol=2 is readable by any Python 3.x version
        pkl_path = os.path.join(output_dir, f"ArcFace_{relation}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(embeddings_array, f, protocol=2)

        # Save filenames separately so you can still trace image → row index
        names_path = os.path.join(output_dir, f"ArcFace_{relation}_filenames.pkl")
        with open(names_path, "wb") as f:
            pickle.dump(filenames, f, protocol=2)

        print(f"  Saved : ArcFace_{relation}.npy   shape={embeddings_array.shape}")
        print(f"  Saved : ArcFace_{relation}.pkl   (plain ndarray, protocol=2)")
        print(f"  Saved : ArcFace_{relation}_filenames.pkl")
        print(f"  Embedding dim : {embeddings_array.shape[1]}")
        print(f"  First file    : {filenames[0]}")
        print(f"  Last file     : {filenames[-1]}")

    print("\nExtraction complete.")