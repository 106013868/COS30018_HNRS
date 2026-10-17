"""
preprocessing.py
loads and caches every dataset used by the project in mnist format
mnist, swiss handwritten digits and handwritten math symbols
run python -m hnrs.preprocessing to rebuild all caches and export the mnist digit images
"""

import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from hnrs.preprocess_image import preprocess_image

PREPROCESS_VERSION = 1 #please update this version number if you change the preprocessing steps

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_MODULE_DIR)
CACHE_PATH = os.path.join(
    os.path.dirname(_MODULE_DIR), "data", f"mnist_preprocessed_v{PREPROCESS_VERSION}.npz"
)

# swiss and math caches are built with preprocess_image
# update these versions if preprocess_image changes so the caches rebuild
SWISS_VERSION = 2
MATH_VERSION = 1

# kaggle: "Handwritten Digits 0 - 9" by olafkrastovski, CC0 licence
# unzip it so the folders 0 to 9 sit inside data/swiss_digits/
SWISS_DIR = os.path.join(_ROOT, "data", "swiss_digits")
SWISS_CACHE_PATH = os.path.join(_ROOT, "data", f"swiss_preprocessed_v{SWISS_VERSION}.npz")

# kaggle: "Handwritten Math Symbols" by sagyamthapa
# unzip it so the dataset folders (0 to 9, add, sub, ...) sit inside data/math_symbols/
MATH_DIR = os.path.join(_ROOT, "data", "math_symbols")
MATH_CACHE_PATH = os.path.join(_ROOT, "data", f"math_preprocessed_v{MATH_VERSION}.npz")

# math dataset folder name -> class the model predicts
# the label is the position in this list, so digits keep labels 0-9
# eq, dec, x, y and z folders are not needed for the brief so they are skipped
# add "(" and ")" here once we have bracket images
MATH_FOLDERS = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "add": "+", "sub": "-", "mul": "*", "div": "/",
}
CLASSES = list(MATH_FOLDERS.values())

# mnist digit images used by compose_numbers.py
DIGITS_DIR = os.path.join(_ROOT, "data", "imgs")


# ---- mnist ----

def load_preprocessed_mnist(cache_path=CACHE_PATH, use_cache=True):
    if use_cache and os.path.exists(cache_path):
        with np.load(cache_path) as data:
            return (
                data["x_train"], data["y_train"], data["y_cat_train"],
                data["x_test"], data["y_test"], data["y_cat_test"],
            )

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_cat_train = to_categorical(y_train, 10)
    y_cat_test = to_categorical(y_test, 10)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        x_train=x_train, y_train=y_train, y_cat_train=y_cat_train,
        x_test=x_test, y_test=y_test, y_cat_test=y_cat_test,
    )

    return x_train, y_train, y_cat_train, x_test, y_test, y_cat_test


def export_mnist_images(per_class=200, output_dir=DIGITS_DIR):
    """
    saves per_class mnist training images of each digit as pngs for compose_numbers.py
    """
    (x_train, y_train), _ = mnist.load_data()

    counts = {}

    for i in range(len(x_train)):

        label = int(y_train[i])
        if counts.get(label, 0) >= per_class:
            continue

        folder = f"{output_dir}/{label}"
        image_name = f"index_{i:05d}.png"

        os.makedirs(folder, exist_ok=True)

        img = Image.fromarray(x_train[i])
        img.save(f"{folder}/{image_name}")

        counts[label] = counts.get(label, 0) + 1

        if len(counts) == 10 and all(c >= per_class for c in counts.values()):
            break


# ---- image folder datasets (swiss, math symbols) ----

def load_image_folders(root, folders):
    """
    In: dataset root folder, list of class folder names
    Out: preprocessed images and labels, label is the folder's position in the list
    """
    images = []
    labels = []
    skipped = 0

    for label, folder in enumerate(folders):
        paths = sorted(glob.glob(os.path.join(root, folder, "*.*")))
        paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not paths:
            raise FileNotFoundError(f"No images found in {root}/{folder}. Unzip the Kaggle dataset there first.")

        for path in paths:
            try:
                images.append(preprocess_image(path))
                labels.append(label)
            except ValueError:
                skipped += 1  # blank or unreadable image

        print(f"{folder}: {len(paths)} images")

    print(f"Preprocessed {len(images)} images, skipped {skipped}")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.uint8)


def split_per_class(x, y, test_size=0.2, seed=42):
    """
    shuffles and holds back test_size of each class, so train and test stay balanced
    returns x_train, y_train, x_test, y_test
    """
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_size)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = rng.permutation(train_idx)
    test_idx = rng.permutation(test_idx)

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def load_cached_split(cache_path, root, folders, use_cache=True, test_size=0.2, seed=42):
    """
    loads the train/test split from cache, or builds it from the image folders and saves it
    """
    if use_cache and os.path.exists(cache_path):
        with np.load(cache_path) as data:
            return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    x, y = load_image_folders(root, folders)
    x_train, y_train, x_test, y_test = split_per_class(x, y, test_size, seed)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    return x_train, y_train, x_test, y_test


def load_swiss_digits(cache_path=SWISS_CACHE_PATH, use_cache=True, test_size=0.2, seed=42):
    """
    In: cache path, whether to use the cache, test split size, seed
    Out: x_train, y_train, x_test, y_test, labels are digits 0-9
    """
    folders = [str(d) for d in range(10)]
    return load_cached_split(cache_path, SWISS_DIR, folders, use_cache, test_size, seed)


def load_math_symbols(cache_path=MATH_CACHE_PATH, use_cache=True, test_size=0.2, seed=42):
    """
    In: cache path, whether to use the cache, test split size, seed
    Out: x_train, y_train, x_test, y_test, labels are positions in CLASSES
    """
    return load_cached_split(cache_path, MATH_DIR, list(MATH_FOLDERS), use_cache, test_size, seed)


if __name__ == "__main__":
    load_preprocessed_mnist(use_cache=False)
    print(f"Preprocessed MNIST cached to {CACHE_PATH}")

    export_mnist_images()
    print(f"MNIST digit images exported to {DIGITS_DIR}")

    load_swiss_digits(use_cache=False)
    print(f"Swiss digits cached to {SWISS_CACHE_PATH}")

    load_math_symbols(use_cache=False)
    print(f"Math symbols cached to {MATH_CACHE_PATH}")
