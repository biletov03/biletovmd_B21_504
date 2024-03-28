import cv2
import numpy as np


def difference_image(img1: np.array, img2: np.array) -> np.array:
    return abs(img1 - img2)


def bitwise_equal_manual(matrix1, matrix2):
    assert len(matrix1) == len(matrix2), "Матрицы должны быть одинакового размера"
    for row1, row2 in zip(matrix1, matrix2):
        assert len(row1) == len(row2), "Матрицы должны быть одинакового размера"

    result_matrix = []
    for row1, row2 in zip(matrix1, matrix2):
        result_row = []
        for bit1, bit2 in zip(row1, row2):
            result_bit = ~(bit1 ^ bit2)
            result_row.append(result_bit)
        result_matrix.append(result_row)

    return result_matrix


def apply_aperture(img, new_image, x, y, size, threshold):
    size //= 2

    left = max(y - size, 0)
    right = min(y + size + 1, img.shape[1])
    low = max(x - size, 0)
    above = min(x + size + 1, img.shape[0])

    aperture = img[low: above, left: right]

    ones = (aperture == 255).sum()

    if ones >= threshold:
        new_image[x, y] = 255


def rank_filter(img, size, rang):
    if size % 2 == 0:
        size += 1

    new_img = np.zeros(shape=img.shape)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            apply_aperture(img, new_img, x, y, size, rang)

    return new_img.astype(np.uint8)


def create_filtered_and_diff_image(start_path, filtered_path, diff_path):
    image = cv2.imread(start_path)

    filtered = rank_filter(image, 5, 20)

    diff = bitwise_equal_manual(filtered, image)

    cv2.imwrite(filtered_path, filtered)

    cv2.imwrite(diff_path, np.array(diff))

#  В файле f_d_i лежат изображения отфильтрованные и их различие с исходным изображением


if __name__ == "__main__":
    images = [["1.jpeg", "2.jpg"], ["1.png", "2.png", "3.png"], ["1.jpeg", "2.jpeg", "3.jpeg"], ["1.png", "2.png","3.png","4.jpg","5.jpeg",]]

    for i in range(1, len(images) + 1):
        for element in images[i - 1]:
            both_part = f"{i}/{element}"
            print(both_part)
            create_filtered_and_diff_image(
                f"bin_imgs/{i}/{element}",
                f"f_d_i/{i}/f_{element}",
                f"f_d_i/{i}/d_{element}"
            )
