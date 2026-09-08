from tensorflow.keras.datasets import mnist
from PIL import Image
import os
import numpy as np

def export_mnist_images(per_class=200, output_dir="data/imgs"):
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

if __name__ == "__main__":
    export_mnist_images()