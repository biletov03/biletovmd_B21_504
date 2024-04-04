import cv2
from PIL import Image
import numpy as np


def KNN_resampling(old_image, scale):
    width = old_image.shape[1]
    height = old_image.shape[0]
    new_width = round(scale * width)
    new_height = round(scale * height)

    new_image = np.zeros(shape=(new_height, new_width, old_image.shape[2]))

    for x in range(new_width):
        for y in range(new_height):
            src_x = min(
                int(round(float(x) / float(new_width) * float(width))), width - 1)
            src_y = min(
                int(round(float(y) / float(new_height) * float(height))), height - 1)

            new_image[y, x] = old_image[src_y, src_x]

    return new_image


def two_iteration(image, extension, compression):
    ex_image = KNN_resampling(image, extension)
    comp_image = KNN_resampling(ex_image, 1 / compression)
    return comp_image


def one_iteration(image, extension, compression):
    res_image = KNN_resampling(image, extension / compression)
    return res_image


if __name__ == '__main__':
    image_names = ['spiral_2.png', 'spiral.png']

    for image_name in image_names:
        img_in_arr = np.array(Image.open(f"input/{image_name}").convert('RGB'))

        one_it_img = one_iteration(img_in_arr, 3, 16)

        two_it_img = two_iteration(img_in_arr, 3, 16)

        cv2.imwrite(f"output/in_one_run/{image_name}", one_it_img)

        cv2.imwrite(f"output/in_two_run/{image_name}", two_it_img)
