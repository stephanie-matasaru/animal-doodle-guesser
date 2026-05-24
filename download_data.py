import os
import urllib.request
import numpy as np

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORIES = [
    "cat", "dog", "elephant", "giraffe", "shark",
    "octopus", "butterfly", "owl", "penguin", "mouse", "snake", "frog",
    "lion", "horse", "rabbit", "pig", "duck", "whale",
    "monkey", "cow", "crocodile", "dolphin", "sheep", "bee",
    "sea turtle", "camel", "snail", "crab"
]

DATA_DIR = "data"
MAX_SAMPLES = 10000  

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    for category in CATEGORIES:
        filename = f"{category}.npy"
        filepath = os.path.join(DATA_DIR, filename)

        if os.path.exists(filepath):
            print(f"[skip] {category} already downloaded")
            continue

        url = BASE_URL + filename.replace(" ", "%20")
        print(f"[download] {category}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"[done] {category}")

def load_data():
    X, y = [], []

    for label, category in enumerate(CATEGORIES):
        filepath = os.path.join(DATA_DIR, f"{category}.npy")
        data = np.load(filepath)         # shape: (N, 784) — flattened 28x28
        data = data[:MAX_SAMPLES]        # limit samples
        X.append(data)
        y.append(np.full(len(data), label))
        print(f"[loaded] {category}: {len(data)} samples")

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Normalize pixel values to [0, 1]
    X = X.astype("float32") / 255.0

    # Reshape to (N, 28, 28, 1) for CNN input
    X = X.reshape(-1, 28, 28, 1)

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    print(f"\nTotal samples: {len(X)}")
    print(f"Shape: {X.shape}")
    return X, y, CATEGORIES


if __name__ == "__main__":
    download_data()
    X, y, categories = load_data()
    print("\nData ready!")
    print(f"Categories: {categories}")