"""
math_symbols.py
loads handwritten digits and arithmetic symbols, preprocessed into mnist format
"""

import os
import glob
import numpy as np

from hnrs.preprocess_image import preprocess_image

MATH_VERSION = 1  # update if preprocessing changes so the cache rebuilds

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_MODULE_DIR)
MATH_DIR = os.path.join(_ROOT, "data", "math_symbols")
CACHE_PATH = os.path.join(_ROOT, "data", f"math_preprocessed_v{MATH_VERSION}.npz")

# kaggle: "Handwritten Math Symbols" by sagyamthapa
# unzip it so the dataset folders (0 to 9, add, sub, ...) sit inside data/math_symbols/

# dataset folder name -> class the model predicts
# the label is the position in this list, so digits keep labels 0-9
# eq, dec, x, y and z folders are not needed for the brief so they are skipped
# add "(" and ")" here once we have bracket images
FOLDERS = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "add": "+", "sub": "-", "mul": "*", "div": "/",
}
CLASSES = list(FOLDERS.values())


def load_math_symbols(cache_path=CACHE_PATH, use_cache=True, test_size=0.2, seed=42):
    """
    In: cache path, whether to use the cache, test split size, seed
    Out: x_train, y_train, x_test, y_test, labels are positions in CLASSES
    """
    if use_cache and os.path.exists(cache_path):
        with np.load(cache_path) as data:
            return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    images = []
    labels = []
    skipped = 0

    for label, folder in enumerate(FOLDERS):
        paths = sorted(glob.glob(os.path.join(MATH_DIR, folder, "*.*")))
        paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not paths:
            raise FileNotFoundError(f"No images found in {MATH_DIR}/{folder}. Unzip the Kaggle dataset there first.")

        for path in paths:
            try:
                images.append(preprocess_image(path))
                labels.append(label)
            except ValueError:
                skipped += 1  # blank image

        print(f"{folder} ({CLASSES[label]}): {len(paths)} images")

    x = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.uint8)
    print(f"Preprocessed {len(x)} images, skipped {skipped}")

    # split each class separately so train and test stay balanced
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for label in range(len(CLASSES)):
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
    x_train, y_train, x_test, y_test = load_math_symbols(use_cache=False)
    print("train:", x_train.shape, "test:", x_test.shape)
    print(f"Cached to {CACHE_PATH}")
