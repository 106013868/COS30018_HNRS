import os
import glob, random
import numpy as np
from PIL import Image

folder = "data/imgs"

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

def load_digit_bank(folder):
    bank = {}

    for path in sorted(glob.glob(folder + "/*")):
        name = os.path.basename(path)
        if name.isdigit():
            bank[int(name)] = sorted(glob.glob(path + "/*.png"))

    if not bank or not all(bank.values()):
        raise FileNotFoundError(f"No images found. Run export_digits.py to generate the digit images.")

    return bank

if __name__ == "__main__":
    bank = load_digit_bank(folder)
    for _ in range(5):
        image, label = generate_number(bank)
        print(label, image.shape)