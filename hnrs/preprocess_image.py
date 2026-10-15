import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage


def preprocess_image(source):
    """Turn a photo or drawing of one digit into a 28x28 image like MNIST.

    MNIST digits are white on black, cropped, fitted into a 20x20 box and
    centred in a 28x28 image. The model only works well if our images
    look the same, so this function copies those steps.

    source can be a file path, a PIL image or a numpy array.
    Returns a 28x28 float32 array with values from 0 to 1.
    """

    # ---- Step 1: load the image ----
    if isinstance(source, np.ndarray):
        img = Image.fromarray(source)
    elif isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)

    # Phone photos are saved sideways with a tag saying how to rotate them.
    # This applies that tag so the digit is upright.
    img = ImageOps.exif_transpose(img)

    # Some PNGs have see-through backgrounds. Put them on white first,
    # otherwise the see-through parts turn black when we convert to greyscale.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)

    # Greyscale, since colour doesn't help recognise a digit.
    img = img.convert("L")

    # Phone photos are huge (around 4000px). We only need 28px in the end,
    # so shrink big images now to make everything after this faster.
    if max(img.size) > 512:
        img.thumbnail((512, 512))

    arr = np.array(img, dtype=np.float32)

    # Small images (MNIST digits, segmentation crops) are already clean.
    # Big images are photos or drawings and need extra cleaning.
    is_photo = max(arr.shape) >= 100

    # ---- Step 2: make it white digit on black background ----
    # The edges of the image are mostly background. If the edges are bright,
    # the background is white paper, so flip the colours.
    edges = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if np.median(edges) > 127:
        arr = 255 - arr

    # ---- Step 3: even out the lighting (photos only) ----
    # Photos are often brighter on one side. Blurring the image heavily
    # gives an estimate of the background at every point. Subtracting it
    # leaves just the ink.
    if is_photo:
        blur = Image.fromarray(arr.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=max(arr.shape) / 8))
        arr = arr - np.array(blur, dtype=np.float32)
        arr = np.clip(arr, 0, None)

    if arr.max() - arr.min() < 30:
        raise ValueError("No digit found in image")

    # ---- Step 4: separate ink from paper ----
    # Anything less than 30% as bright as the brightest ink counts as
    # background and gets set to 0.
    arr[arr < 0.3 * arr.max()] = 0

    # ---- Step 5: remove specks ----
    # Dust, paper texture and shadows leave small blobs of "ink". If we keep
    # them, the crop in the next step gets too big and the digit ends up tiny.
    # label() gives every separate blob its own number so we can check each one.
    blobs, num_blobs = ndimage.label(arr > 0)

    if num_blobs > 1:
        keep = list(range(1, num_blobs + 1))

        # In a photo, shadows and paper edges show up as blobs along the border.
        # Drop blobs that touch the edge AND sit near it. A digit whose tail
        # just reaches the edge is still centred, so it gets kept.
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

        # Drop blobs smaller than 15% of the biggest one. This clears dust
        # but keeps parts of a digit that don't join up, like the top of a 5.
        sizes = {b: (blobs == b).sum() for b in keep}
        biggest = max(sizes.values())
        keep = [b for b in keep if sizes[b] >= 0.15 * biggest]

        arr[~np.isin(blobs, keep)] = 0

    # ---- Step 6: crop to the digit ----
    rows = np.where(arr.any(axis=1))[0]
    cols = np.where(arr.any(axis=0))[0]
    arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # ---- Step 7: thicken thin pen strokes ----
    # A thin pen line almost disappears when shrunk to 20px. Resize to 100px
    # first, measure how thick the strokes are, and thicken them if needed.
    digit = Image.fromarray((arr / arr.max() * 255).astype(np.uint8))
    digit.thumbnail((100, 100), Image.BOX)
    if max(digit.size) < 100:
        scale = 100 / max(digit.size)
        digit = digit.resize((round(digit.width * scale), round(digit.height * scale)), Image.BOX)

    # Rough stroke thickness = ink area / edge length * 2.
    # (A line 10px wide has 2 edges, so area / edges gives half its width.)
    ink = np.array(digit) > 127
    edge = ink & ~ndimage.binary_erosion(ink)
    thickness = 2 * ink.sum() / max(edge.sum(), 1)

    # MNIST strokes are about 2px in a 20px box, which is about 10px at this size.
    # MaxFilter makes every white pixel spread to its neighbours.
    missing = round(10 - thickness)
    if missing >= 2:
        size = missing // 2 * 2 + 1  # MaxFilter needs an odd number
        digit = digit.filter(ImageFilter.MaxFilter(size))

    # ---- Step 8: fit the digit into a 20x20 box ----
    # Keep the shape, so a 1 stays thin instead of getting stretched.
    scale = 20 / max(digit.size)
    new_w = max(1, round(digit.width * scale))
    new_h = max(1, round(digit.height * scale))
    digit = np.array(digit.resize((new_w, new_h), Image.BOX), dtype=np.float32)

    # ---- Step 9: centre it on a 28x28 black image ----
    # MNIST centres digits by their centre of mass (the "balance point" of
    # the ink), not by the middle of the box, so we do the same.
    cy, cx = ndimage.center_of_mass(digit)
    top = int(np.clip(round(14 - cy), 0, 28 - new_h))
    left = int(np.clip(round(14 - cx), 0, 28 - new_w))

    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[top:top + new_h, left:left + new_w] = digit

    # ---- Step 10: scale to 0-1, like the training data ----
    return canvas / canvas.max()
