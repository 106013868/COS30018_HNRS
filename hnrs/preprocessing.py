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

SWISS_VERSION = 3
MATH_VERSION = 3

# kaggle: "Handwritten Digits 0 - 9"
SWISS_DIR = os.path.join(_ROOT, "data", "swiss_digits")
SWISS_CACHE_PATH = os.path.join(_ROOT, "data", f"swiss_preprocessed_v{SWISS_VERSION}.npz")

# kaggle: "Handwritten Math Symbols"
MATH_DIR = os.path.join(_ROOT, "data", "math_symbols")
MATH_CACHE_PATH = os.path.join(_ROOT, "data", f"math_preprocessed_v{MATH_VERSION}.npz")

CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")"]

# dataset folder name -> class in CLASSES
SWISS_FOLDERS = {str(d): str(d) for d in range(10)}

# eq, dec, x, y and z folders are not needed for the brief so they are skipped
MATH_FOLDERS = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "add": "+", "sub": "-", "mul": "*", "div": "/",
    "left_bracket": "(", "right_bracket": ")",
}

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
    In: dataset root folder, dict of folder name -> class in CLASSES
    Out: preprocessed images and labels, label is the class's position in CLASSES
    """
    images = []
    labels = []
    skipped = 0

    for folder, char in folders.items():
        label = CLASSES.index(char)
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
    return load_cached_split(cache_path, SWISS_DIR, SWISS_FOLDERS, use_cache, test_size, seed)


def load_math_symbols(cache_path=MATH_CACHE_PATH, use_cache=True, test_size=0.2, seed=42):
    return load_cached_split(cache_path, MATH_DIR, MATH_FOLDERS, use_cache, test_size, seed)


# ---- dataset registry ----

def load_mnist_split():
    """
    returns mnist as x_train, y_train, x_test, y_test, the same format as the other datasets
    """
    x_train, y_train, _, x_test, y_test, _ = load_preprocessed_mnist()
    return x_train, y_train, x_test, y_test

DATASETS = {
    "mnist": load_mnist_split,
    "swiss": load_swiss_digits,
    "math": load_math_symbols,
}

def load_dataset(name):
    """
    In: dataset name from DATASETS
    Out: x_train, y_train, x_test, y_test
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}', choose from {list(DATASETS)}")
    return DATASETS[name]()


if __name__ == "__main__":
    load_preprocessed_mnist(use_cache=False)
    print(f"Preprocessed MNIST cached to {CACHE_PATH}")

    export_mnist_images()
    print(f"MNIST digit images exported to {DIGITS_DIR}")

    load_swiss_digits(use_cache=False)
    print(f"Swiss digits cached to {SWISS_CACHE_PATH}")

    load_math_symbols(use_cache=False)
    print(f"Math symbols cached to {MATH_CACHE_PATH}")
