"""
preprocess_image.py
converts a photo or drawing of a single digit into mnist format for the model
"""

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage


def preprocess_image(source):
    """
    In: file path, PIL image or numpy array of one digit
    Out: 28x28 float32 array scaled 0-1, white digit on black like mnist
    """

    # load image from any supported input type
    if isinstance(source, np.ndarray):
        img = Image.fromarray(source)
    elif isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)

    # phone photos store rotation in exif, apply it so digit is upright
    img = ImageOps.exif_transpose(img)

    # put transparent pngs on white, otherwise transparent areas go black in greyscale
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)

    # greyscale, colour doesnt help recognise digits
    img = img.convert("L")

    # shrink large photos early to speed up later steps
    if max(img.size) > 512:
        img.thumbnail((512, 512))

    arr = np.array(img, dtype=np.float32)

    # small images (mnist, segmentation crops) are already clean, large ones need extra cleaning
    is_photo = max(arr.shape) >= 100

    # invert if background is white, checked using median of the border pixels
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if np.median(edges) > 127:
        arr = 255 - arr

    # remove uneven lighting from photos
    # heavy blur estimates the background, subtracting it leaves the ink
    if is_photo:
        blur = Image.fromarray(arr.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=max(arr.shape) / 8))
        arr = arr - np.array(blur, dtype=np.float32)
        arr = np.clip(arr, 0, None)

    if arr.max() - arr.min() < 30:
        raise ValueError("No digit found in image")

    # anything under 30% of the brightest pixel counts as background
    arr[arr < 0.3 * arr.max()] = 0

    # remove specks so the crop isnt stretched by dust or shadows
    # label() gives each separate blob its own number
    blobs, num_blobs = ndimage.label(arr > 0)

    if num_blobs > 1:
        keep = list(range(1, num_blobs + 1))

        # photos only: drop blobs touching the border unless their centre is in the middle
        # keeps digits whose tail reaches the edge, removes shadows and paper edges
        if is_photo:
            edge_blobs = set(np.concatenate([blobs[0], blobs[-1], blobs[:, 0], blobs[:, -1]]))
            h, w = arr.shape
            middle = []
            for b in keep:
                if b in edge_blobs:
                    cy, cx = ndimage.center_of_mass(blobs == b)
                    near_edge = not (h * 0.25 < cy < h * 0.75 and w * 0.25 < cx < w * 0.75)
                    if near_edge:
                        continue
                middle.append(b)
            if middle:
                keep = middle

        sizes = {b: (blobs == b).sum() for b in keep}
        biggest = max(sizes, key=sizes.get)

        # rows and columns covered by the biggest blob, used to find the dots in ÷
        main_rows = np.where((blobs == biggest).any(axis=1))[0]
        main_cols = np.where((blobs == biggest).any(axis=0))[0]

        def is_dot(b):
            """
            returns True if blob b sits directly above or below the biggest blob
            """
            cy, cx = ndimage.center_of_mass(blobs == b)
            above_or_below = cy < main_rows[0] or cy > main_rows[-1]
            in_line = main_cols[0] <= cx <= main_cols[-1]
            return sizes[b] >= 0.02 * sizes[biggest] and above_or_below and in_line

        # drop blobs under 15% of the biggest one
        # clears dust but keeps disconnected digit parts like the top of a 5, and ÷ dots
        keep = [b for b in keep if sizes[b] >= 0.15 * sizes[biggest] or is_dot(b)]

        arr[~np.isin(blobs, keep)] = 0

    # crop to the digit
    rows = np.where(arr.any(axis=1))[0]
    cols = np.where(arr.any(axis=0))[0]
    arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # resize to 100px so stroke thickness can be measured before shrinking to 20px
    # thin pen lines would mostly disappear otherwise
    digit = Image.fromarray((arr / arr.max() * 255).astype(np.uint8))
    digit.thumbnail((100, 100), Image.BOX)
    if max(digit.size) < 100:
        scale = 100 / max(digit.size)
        digit = digit.resize((round(digit.width * scale), round(digit.height * scale)), Image.BOX)

    # estimate stroke thickness as 2 * ink area / edge length
    # a line has 2 edges so area / edges is half its width
    ink = np.array(digit) > 127
    edge = ink & ~ndimage.binary_erosion(ink)
    thickness = 2 * ink.sum() / max(edge.sum(), 1)

    # mnist strokes are about 10px at this size, thicken if ours are thinner
    # MaxFilter spreads each white pixel to its neighbours
    missing = round(10 - thickness)
    if missing >= 2:
        size = missing // 2 * 2 + 1  # MaxFilter needs an odd size
        digit = digit.filter(ImageFilter.MaxFilter(size))

    # fit into 20x20 box, keeping aspect ratio so a 1 stays thin
    scale = 20 / max(digit.size)
    new_w = max(1, round(digit.width * scale))
    new_h = max(1, round(digit.height * scale))
    digit = np.array(digit.resize((new_w, new_h), Image.BOX), dtype=np.float32)

    # place on 28x28 canvas centred by centre of mass, same as mnist
    cy, cx = ndimage.center_of_mass(digit)
    top = int(np.clip(round(14 - cy), 0, 28 - new_h))
    left = int(np.clip(round(14 - cx), 0, 28 - new_w))

    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[top:top + new_h, left:left + new_w] = digit

    # scale to 0-1 to match training data
    return canvas / canvas.max()
