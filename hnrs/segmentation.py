"""
segmentation.py
splits an image of a number or expression into one image per character, left to right
finds each blob of ink, then joins blobs that sit above or below each other into one character
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage


def split_characters(source):
    """
    In: file path, PIL image or numpy array of a row of handwritten characters
    Out: list of greyscale crops, one per character, ready for preprocess_image
    """
    # load as greyscale, same as preprocess_image
    if isinstance(source, np.ndarray):
        img = Image.fromarray(source)
    elif isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    img = img.convert("L")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024))
    grey = np.array(img)

    # ink mask: white ink on black, with uneven lighting removed from photos
    arr = grey.astype(np.float32)
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if np.median(edges) > 127:
        arr = 255 - arr
    if min(arr.shape) >= 100:
        blur = Image.fromarray(arr.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=min(arr.shape) / 8))
        arr = np.clip(arr - np.array(blur, dtype=np.float32), 0, None)
    if arr.max() < 30:
        return []
    ink = arr > 0.3 * arr.max()

    # each separate blob of ink, left to right
    # thin pen strokes can break into pieces, so close gaps of a few pixels first
    rows = np.where(ink.any(axis=1))[0]
    gap = max(1, round(0.03 * (rows[-1] - rows[0] + 1)))
    joined = ndimage.binary_dilation(ink, iterations=gap)
    blobs, count = ndimage.label(joined, structure=np.ones((3, 3)))
    blobs[~ink] = 0
    boxes = ndimage.find_objects(blobs)
    order = sorted(range(count), key=lambda i: boxes[i][1].start)

    # blobs that sit above or below each other belong to the same character,
    # e.g. the dots and bar in a divide sign, or the two strokes of a 5
    groups = []
    for i in order:
        x0, x1 = boxes[i][1].start, boxes[i][1].stop
        if groups:
            g = groups[-1]
            overlap = min(x1, g["x1"]) - max(x0, g["x0"])
            if overlap >= 0.5 * min(x1 - x0, g["x1"] - g["x0"]):
                g["blobs"].append(i + 1)
                g["x0"], g["x1"] = min(x0, g["x0"]), max(x1, g["x1"])
                continue
        groups.append({"blobs": [i + 1], "x0": x0, "x1": x1})

    # drop groups with hardly any ink, e.g. dust between characters
    for g in groups:
        g["mask"] = np.isin(blobs, g["blobs"])
        g["ink"] = g["mask"].sum()
    groups = [g for g in groups if g["ink"] >= 0.01 * max(g["ink"] for g in groups)]

    # background colour, used to blank out anything that isn't this character
    background = np.median(np.concatenate([grey[0], grey[-1], grey[:, 0], grey[:, -1]]))
    all_ink = ndimage.binary_dilation(ink, iterations=2)

    crops = []
    for g in groups:
        rows = np.where(g["mask"].any(axis=1))[0]
        cols = np.where(g["mask"].any(axis=0))[0]
        top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1

        # crop with a margin so the character doesn't touch the edges
        pad = max(2, int(0.2 * max(bottom - top, right - left)))
        y0, x0 = max(0, top - pad), max(0, left - pad)
        crop = grey[y0:bottom + pad, x0:right + pad].copy()

        # the margin can catch parts of other characters, so fill them with background
        own = ndimage.binary_dilation(g["mask"], iterations=2)[y0:bottom + pad, x0:right + pad]
        other = all_ink[y0:bottom + pad, x0:right + pad] & ~own
        crop[other] = background
        crops.append(crop)

    return crops