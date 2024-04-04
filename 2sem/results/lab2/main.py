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


def create_bin_image(start_img_path: str, result_img_path: str) -> None:
    image = cv2.imread(start_img_path)

    binary_image = balanced_thresholding(image)

    cv2.imwrite(result_img_path, binary_image)


if __name__ == "__main__":
    images = [
        ["1.jpeg", "2.jpg"],
        ["1.png", "2.png", "3.png"],
        ["1.jpeg", "2.jpeg", "3.jpeg"],
        ["1.png", "2.png", "3.png", "4.jpg", "5.jpeg"]
    ]

    for i in range(1, len(images) + 1):
        for element in images[i - 1]:
            both_part = f"{i}/{element}"
            create_bin_image(f"input/{both_part}", f"output/{both_part}")
