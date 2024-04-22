import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from helper import FeatureImage

input_symbols = [
    '0A73.png', '0A05.png', '0A72.png', '0A38.png', '0A39.png', '0A18.png', '0A19.png', '0A1A.png', '0A1B.png',
    '0A1C.png', '0A20.png', '0A21.png', '0A22.png', '0A15.png', '0A16.png', '0A17.png', '0A1D.png', '0A1E.png',
    '0A1F.png', '0A23.png', '0A24.png', '0A25.png', '0A26.png', '0A27.png', '0A28.png', '0A2A.png', '0A2B.png',
    '0A2C.png', '0A2D.png', '0A2E.png', '0A2F.png', '0A30.png', '0A32.png', '0A35.png', '0A5C.png', '0A36.png',
    '0A59.png', '0A5A.png', '0A5B.png', '0A5E.png', '0A33.png', '0A05_1.png', '0A06.png', '0A07.png', '0A08.png',
    '0A09.png', '0A0A.png', '0A0F.png', '0A10.png', '0A74.png', '0A13.png', '0A14.png', '0A71.png', '0A02.png',
    '0A70.png', '0A03.png', '0A4D.png'
]


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bin_img


def save_features_to_csv(features, image_name):
    df = pd.DataFrame([features])
    df.to_csv(f'{image_name}_features.csv', sep=',', index=False)


def plot_and_save_histogram(data, image_name, title):
    plt.figure(figsize=(10, 6))

    plt.bar(np.arange(data.size), data)

    plt.title(title)

    plt.xlabel('Pixel Position')
    plt.ylabel('Count')

    x_ticks_interval = data.size // 10
    y_ticks_interval = max(data) // 10
    plt.xticks(np.arange(0, data.size, x_ticks_interval))
    plt.yticks(np.arange(0, max(data) + 1, y_ticks_interval))

    plt.tight_layout()

    plt.savefig(f'{image_name}_{title}.png')

    plt.close()


def calculate_features(img):
    feature_img = FeatureImage(img)

    quarters_mass = [
        feature_img.weight_I(),
        feature_img.weight_II(),
        feature_img.weight_III(),
        feature_img.weight_IV()
    ]

    quarters_weight = [
        feature_img.relative_weight_I(),
        feature_img.relative_weight_II(),
        feature_img.relative_weight_III(),
        feature_img.relative_weight_IV()
    ]

    centroid_x = feature_img.center(1)
    centroid_y = feature_img.center(0)

    centroid_x_norm = feature_img.relative_center(1)
    centroid_y_norm = feature_img.relative_center(0)

    mu20 = feature_img.inertia(1)
    mu02 = feature_img.inertia(0)

    mu20_norm = feature_img.relative_inertia(1)
    mu02_norm = feature_img.relative_inertia(0)

    profile_x = np.sum(img == 0, axis=0)
    profile_y = np.sum(img == 0, axis=1)

    features = {
        'quarters_mass': quarters_mass,
        'quarters_weight': quarters_weight,
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'centroid_x_norm': centroid_x_norm,
        'centroid_y_norm': centroid_y_norm,
        'mu20': mu20,
        'mu02': mu02,
        'mu20_norm': mu20_norm,
        'mu02_norm': mu02_norm,
    }

    return features, profile_x, profile_y


def process_images(image_names):
    for image_name in image_names:
        try:
            image_name = image_name.split(".")[0]
            image_path = f"input/{image_name}/{image_name}.png"
            bin_img = load_image(image_path)
            features, profile_x, profile_y = calculate_features(bin_img)
            save_features_to_csv(features, image_name)
            plot_and_save_histogram(profile_x, image_name, 'profile_x')
            plot_and_save_histogram(profile_y, image_name, 'profile_y')

            os.system(f"mv {image_name}_features.csv output/{image_name}/features.csv")
            os.system(f"mv {image_name}_profile_x.png output/{image_name}/profile_x.png")
            os.system(f"mv {image_name}_profile_y.png output/{image_name}/profile_y.png")
        except Exception as e:
            print(image_name, e)


process_images(input_symbols)
