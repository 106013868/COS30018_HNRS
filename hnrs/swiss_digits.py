import os
import glob
import numpy as np

from hnrs.preprocess_image import preprocess_image

SWISS_VERSION = 1 #please update this version number if you change the preprocessing steps

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_MODULE_DIR)
SWISS_DIR = os.path.join(_ROOT, "data", "swiss_digits")
CACHE_PATH = os.path.join(_ROOT, "data", f"swiss_preprocessed_v{SWISS_VERSION}.npz")

# Kaggle: "Handwritten Digits 0 - 9" by olafkrastovski, CC0 licence.
# Unzip it so the folders 0 to 9 sit inside data/swiss_digits/


def load_swiss_digits(cache_path=CACHE_PATH, use_cache=True, test_size=0.2, seed=42):
    """Load the Swiss handwritten digits, preprocessed into MNIST format.

    The first run preprocesses all ~21,500 images and saves them to a cache
    file, which takes a few minutes. Later runs load the cache instantly.

    Returns x_train, y_train, x_test, y_test.
    x is (n, 28, 28) float32 from 0 to 1, y is the integer label 0 to 9.
    """
    if use_cache and os.path.exists(cache_path):
        with np.load(cache_path) as data:
            return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    images = []
    labels = []
    skipped = 0

    for label in range(10):
        paths = sorted(glob.glob(os.path.join(SWISS_DIR, str(label), "*.jpg")))
        if not paths:
            raise FileNotFoundError(f"No images found in {SWISS_DIR}/{label}. Unzip the Kaggle dataset there first.")

        for path in paths:
            try:
                images.append(preprocess_image(path))
                labels.append(label)
            except ValueError:
                skipped += 1  # blank or unreadable image

        print(f"digit {label}: {len(paths)} images")

    x = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.uint8)
    print(f"Preprocessed {len(x)} images, skipped {skipped}")

    # Shuffle with a fixed seed, then hold back test_size of each digit for testing.
    # Splitting per digit keeps both halves balanced.
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for label in range(10):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_size)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = rng.permutation(train_idx)
    test_idx = rng.permutation(test_idx)

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_swiss_digits(use_cache=False)
    print("train:", x_train.shape, "test:", x_test.shape)
    print(f"Cached to {CACHE_PATH}")
