import cv2
import numpy as np
import matplotlib.pyplot as plt
from hnrs.preprocess_image import load_image, prepare_page, crop_with_margin, tune_seg


def segment_components(binary, min_area=30, gray=None):
    # thicken ink so small gaps inside a digit close up
    count, label_map, stats = tune_seg(binary)
    ink = binary if gray is None else 255 - gray

    num = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        only_this = np.where(label_map == i, 255, 0).astype(np.uint8) #helps to separate two blobs from connecting
        num.append((x, crop_with_margin(only_this, ink, (x, x + w, y, y + h))))
    #     plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, color="red"))
    # plt.axis("off")
    # plt.show()

    num.sort(key=lambda b: b[0])
    return [crop for x, crop in num]


def segment(img):
    gray = prepare_page(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return segment_components(cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15), gray=gray)


def split_characters(source):
    gray = np.array(load_image(source))
    return segment(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


def main():
    img = cv2.imread("data/Test data for img segmentation/7062543-0155-01.png")
    digits = segment(img)
    for digit in digits:
        plt.imshow(digit, cmap="gray")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()