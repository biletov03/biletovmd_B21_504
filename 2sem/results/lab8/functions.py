import cv2
import numpy as np
from PIL import Image
from numpy import log
from matplotlib import pyplot as plt


def calculate_haralick_features_from_glcm(glcm):
    features = {}
    asm = np.square(glcm).sum()
    features['Asm'] = asm
    max_prob = np.max(glcm)
    features['Mpr'] = max_prob
    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
    features['Ent'] = entropy
    trace = np.trace(glcm)
    features['Tr'] = trace

    return features


def calculate_glcm(image, distance=1, levels=256):
        matrix = np.zeros(shape=(256, 256))

        for x in range(distance, image.shape[0] - distance):
            for y in range(distance, image.shape[1] - distance):
                matrix[image[x - distance, y], image[x, y]] += 1
                matrix[image[x + distance, y], image[x, y]] += 1
                matrix[image[x, y - distance], image[x, y]] += 1
                matrix[image[x, y + distance], image[x, y]] += 1

        for x in range(levels):
            m = np.array(matrix[x])
            m[np.where(m == 0)] = 1
            matrix[x] = log(m)
        matrix = matrix * levels / np.max(matrix)
        return matrix


def plot_glcm(glcm):
    plt.figure(figsize=(10, 10))
    plt.imshow(glcm, interpolation='nearest', cmap='gray')
    plt.title("Gray-Level Co-occurrence Matrix")
    plt.colorbar()
    plt.show()


def gamma_correction(image, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def semitone(img):
    return (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 *
            img[:, :, 2]).astype(np.uint8)


def to_semitone(img_name):
    img = image_to_np_array(img_name)
    return Image.fromarray(semitone(img), 'L')


def image_to_np_array(image_name: str) -> np.array:
    img_src = Image.open(image_name).convert('RGB')
    return np.array(img_src)
