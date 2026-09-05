import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_MODULE_DIR, "data", "mnist_preprocessed.npz")


def load_preprocessed_mnist(cache_path=CACHE_PATH, use_cache=True):
    
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return (
            data["x_train"], data["y_train"], data["y_cat_train"],
            data["x_test"], data["y_test"], data["y_cat_test"],
        )

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_cat_train = to_categorical(y_train)
    y_cat_test = to_categorical(y_test)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        x_train=x_train, y_train=y_train, y_cat_train=y_cat_train,
        x_test=x_test, y_test=y_test, y_cat_test=y_cat_test,
    )

    return x_train, y_train, y_cat_train, x_test, y_test, y_cat_test


if __name__ == "__main__":
    load_preprocessed_mnist(use_cache=False)
    print(f"Preprocessed MNIST cached to {CACHE_PATH}")
