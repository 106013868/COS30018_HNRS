import glob, random
import numpy as np
from PIL import Image

folder = "data/imgs"

bank = {}

def generate_number():
    for i in range(len(glob.glob(folder + "/*"))):
        bank[i] = sorted(glob.glob(folder + f"/{i}/*.png"))

    digits = []
    length = random.randint(1, 4)

    for i in range(length):
        digits.append(random.randint(0, 9))

    parts = []

    for i, digit in enumerate(digits):
        parts.append(np.array(Image.open(random.choice(bank[digit]))))

        if i < len(digits) - 1:
            gap = random.randint(0, 10)
            parts.append(np.zeros((28, gap), dtype=np.uint8))

    result = np.hstack(parts)

    result_image = Image.fromarray(result)
    result_image.save("result.png")

if __name__ == "__main__":
    generate_number()