import cv2
import numpy as np


def rgb_to_gray(rgb_image: np.array) -> np.array:
    weights = np.array([0.299, 0.587, 0.114])  # Веса для RGB каналов согласно стандарту
    gray_image = np.dot(rgb_image[..., :3], weights).astype(np.uint8)
    return gray_image


def balanced_thresholding(image: np.array) -> np.array:
    gray = rgb_to_gray(image)
    threshold = np.mean(gray)
    while True:
        # Разделение пикселей на основе текущего порога
        foreground = gray[gray > threshold]
        background = gray[gray <= threshold]

        if len(foreground) == 0 or len(background) == 0:
            break  # / 0
        new_threshold = (np.mean(foreground) + np.mean(background)) / 2

        if abs(new_threshold - threshold) < 0.001:
            break
        threshold = new_threshold

    _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return binary_image


def dilate_black_regions(image, kernel):
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                          constant_values=255)

    modified_image = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            section = padded_image[y:y + kernel.shape[0], x:x + kernel.shape[1]]

            min_value = np.min(section * kernel)

            modified_image[y, x] = 0 if min_value == 0 else 255

    return modified_image


def create_bin_image(start_img_path: str):
    image = cv2.imread(start_img_path)

    binary_image = balanced_thresholding(image)

    return binary_image


if __name__ == "__main__":
    input_files = ["flight.jpg", "fon.jpg", "keyboard.png", "stair.jpg", "table.jpg", "wall.jpg"]
    for img_puth in input_files:
        bin_img = create_bin_image("input/" + img_puth)

        kernel = np.ones((3, 3), dtype=np.uint8)

        res = dilate_black_regions(bin_img, kernel)

        cv2.imwrite("output/bin/bin_" + img_puth, bin_img)
        cv2.imwrite("output/dilate/dilate_" + img_puth, res)

        difference_image = bin_img - res

        cv2.imwrite("output/contur/contur_" + img_puth, difference_image)
