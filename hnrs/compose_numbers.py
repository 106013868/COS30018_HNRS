import csv
import os
import glob, random
import numpy as np
from PIL import Image

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_MODULE_DIR)
DIGITS_DIR = os.path.join(_ROOT, "data", "imgs")
OUTPUT_DIR = os.path.join(_ROOT, "data", "generated")

def generate_number(bank, min_len=1, max_len=4, min_gap=0, max_gap=10):
    digits = []
    length = random.randint(min_len, max_len)

    for i in range(length):
        digits.append(random.randint(0, 9))

    parts = []

    for i, digit in enumerate(digits):
        parts.append(np.array(Image.open(random.choice(bank[digit]))))

        if i < len(digits) - 1:
            gap = random.randint(min_gap, max_gap)
            parts.append(np.zeros((28, gap), dtype=np.uint8))

    image = np.hstack(parts)
    label = "".join(map(str, digits))

    return image, label

def generate_batch(bank, count, out_dir=OUTPUT_DIR, min_len=1, max_len=4, min_gap=0, max_gap=10):
    os.makedirs(out_dir, exist_ok=True)
    # remove old images
    for old in glob.glob(os.path.join(out_dir, "*.png")):
        os.remove(old)

    # Create a CSV file to store the labels
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        for i in range(count):
            image, label = generate_number(bank, min_len=min_len, max_len=max_len, min_gap=min_gap, max_gap=max_gap)
            filename = f"{i:04d}_{label}.png"

            Image.fromarray(image).save(os.path.join(out_dir, filename))
            writer.writerow([filename, label])


def load_digit_bank(folder=DIGITS_DIR):
    bank = {}

    for path in sorted(glob.glob(folder + "/*")):
        name = os.path.basename(path)
        if name.isdigit():
            bank[int(name)] = sorted(glob.glob(path + "/*.png"))

    if not bank or not all(bank.values()):
        raise FileNotFoundError(f"No images found in {folder}. Run export_digits.py to generate the digit images.")

    return bank

if __name__ == "__main__":
    bank = load_digit_bank()
    generate_batch(bank, 20)