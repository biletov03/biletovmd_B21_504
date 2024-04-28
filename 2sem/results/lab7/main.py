import json

import cv2
from helper import FeatureImage
import csv

from lab7.segmentator import process_images


def dist(vector1, vector2):
    assert len(vector1) == len(vector2)
    sum_square_diff = 0

    for coord1, coord2 in zip(vector1, vector2):
        sum_square_diff += (coord1 - coord2) ** 2
    return sum_square_diff


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bin_img


def save_to_csv(letter_dict, csv_file_name):
    max_pairs = max(len(values) for values in letter_dict.values())

    headers = ['LetterID']
    for i in range(max_pairs):
        headers.append(f'Symbol_{i + 1}')
        headers.append(f'Proximity_{i + 1}')

    with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for letter, symbols_data in letter_dict.items():
            row = [letter]
            for symbol, proximity in symbols_data:
                row.append(symbol)
                row.append(proximity)
            writer.writerow(row)


if __name__ == '__main__':
    input_symbols = [
        '0A73', '0A05', '0A72', '0A38', '0A39', '0A18', '0A19', '0A1A', '0A1B',
        '0A1C', '0A20', '0A21', '0A22', '0A15', '0A16', '0A17', '0A1D', '0A1E',
        '0A1F', '0A23', '0A24', '0A25', '0A26', '0A27', '0A28', '0A2A', '0A2B',
        '0A2C', '0A2D', '0A2E', '0A2F', '0A30', '0A32', '0A35', '0A5C', '0A36',
        '0A59', '0A5A', '0A5B', '0A5E', '0A33', '0A05_1', '0A06', '0A07', '0A08',
        '0A09', '0A0A', '0A0F', '0A10', '0A74', '0A13', '0A14', '0A71', '0A02',
        '0A70', '0A03', '0A4D'
    ]
    features = {}

    count_dict = {"wide": 32, "base": 33}
    for image_name in input_symbols:
        bin_img = load_image(f'../lab5/input/{image_name}/{image_name}.png')
        symbol = FeatureImage(bin_img)
        feature_vector = [
            symbol.relative_weight_I(),
            symbol.relative_weight_II(),
            symbol.relative_weight_III(),
            symbol.relative_weight_IV(),
            symbol.relative_center(1),
            symbol.relative_center(0),
            symbol.relative_inertia(1),
            symbol.relative_inertia(0)
        ]
        features[image_name] = list(feature_vector)
    image_paths = ["base", "wide"]

    for path in image_paths:
        process_images(path)

        letter_dict = {}

        for letter in range(count_dict[path]):
            letter_dict[letter] = []
            symbol = FeatureImage(load_image(f'input/{path}/{letter}.png'))
            feature_vector = [
                symbol.relative_weight_I(),
                symbol.relative_weight_II(),
                symbol.relative_weight_III(),
                symbol.relative_weight_IV(),
                symbol.relative_center(1),
                symbol.relative_center(0),
                symbol.relative_inertia(1),
                symbol.relative_inertia(0)
            ]

            max_dist = 0
            for symbol in input_symbols:
                distance = dist(feature_vector, features[symbol])
                max_dist = max(max_dist, distance)

            for symbol in input_symbols:
                distance = dist(feature_vector, features[symbol])
                proximit = 1 - distance / max_dist
                letter_dict[letter].append([symbol, proximit])
            letter_dict[letter].sort(key=lambda x: x[1], reverse=True)

        file_name = f'output/{path}_letter_features.csv'
        save_to_csv(letter_dict, file_name)
