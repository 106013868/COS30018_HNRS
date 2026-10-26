import cv2
import numpy as np
import matplotlib.pyplot as plt


def segment_components(binary, min_area=30):
    # thicken ink so small gaps inside a digit close up
    kernel = np.ones((3, 3), np.uint8)
    thick = cv2.dilate(binary, kernel)


    count, label_map, stats, centroids = cv2.connectedComponentsWithStats(thick)

    num = []
    for i in range(1, count):                 
        x, y, w, h, area = stats[i]
        if area < min_area:                   
            continue
        
        only_this = np.where(label_map == i, 255, 0).astype(np.uint8) #helps to separate two blobs from connecting
        num.append((x, only_this[y:y + h, x:x + w]))
    #     plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, color="red"))
    # plt.axis("off")
    # plt.show()

    num.sort(key=lambda b: b[0])       
    return [crop for x, crop in num]

def segment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return segment_components(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15))


def main():
    img = cv2.imread("data/Test data for img segmentation/7062543-0155-01.png")
    digits = segment(img)
    for digit in digits:
        plt.imshow(digit, cmap="gray")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
