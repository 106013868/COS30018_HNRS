from tensorflow.keras.datasets import mnist
from PIL import Image
import os

def export_mnist_images():
    (x_train, y_train), _ = mnist.load_data()

    per_class = 200
    counts = {}

    for i in range(len(x_train)):
        label = y_train[i]
        if counts.get(label, 0) >= per_class:
            continue

        dir = f"data/imgs/{label}"
        image_name = f"index_{i:05d}.png"

        os.makedirs(dir, exist_ok=True)

        img = Image.fromarray(x_train[i])
        img.save(f"{dir}/{image_name}")

        counts[label] = counts.get(label, 0) + 1

if __name__ == "__main__":
    export_mnist_images()