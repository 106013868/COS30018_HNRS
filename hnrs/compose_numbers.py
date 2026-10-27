import os
import glob, random
import subprocess
import sys
import os
from collections import Counter
try:
    import numpy as np
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pillow"])
    import numpy as np
    from PIL import Image

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_MODULE_DIR)
DIGITS_DIR = os.path.join(_ROOT, "data", "imgs")

def generate_number(bank, digits=None, min_len=1, max_len=4, min_gap=0, max_gap=10):
    if digits is None:
        length = random.randint(min_len, max_len)
        digits = [random.randint(0, 9) for _ in range(length)]
    else:
        if isinstance(digits, str):
            if not digits.isdigit():
                raise ValueError(f"digits must contain only 0-9, got '{digits}'")
            digits = [int(c) for c in digits]

        missing = [d for d in digits if d not in bank]
        if missing:
            raise ValueError(f"No images available for digit(s) {missing}")

    parts = []

    for i, digit in enumerate(digits):
        parts.append(np.array(Image.open(random.choice(bank[digit]))))

        if i < len(digits) - 1:
            gap = random.randint(min_gap, max_gap)
            parts.append(np.zeros((28, gap), dtype=np.uint8))

    image = np.hstack(parts)
    label = "".join(map(str, digits))

    return image, label

def load_digit_bank(folder=DIGITS_DIR):
    bank = {}

    for path in sorted(glob.glob(folder + "/*")):
        name = os.path.basename(path)
        if name.isdigit():
            bank[int(name)] = sorted(glob.glob(path + "/*.png"))

    if not bank or not all(bank.values()):
        raise FileNotFoundError(f"No images found in {folder}. Run python -m hnrs.preprocessing to generate the digit images.")

    return bank