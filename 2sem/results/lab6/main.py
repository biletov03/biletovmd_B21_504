import cv2
import numpy as np
from matplotlib import pyplot as plt

from lab6.report_maker import generate_markdown_report


def plot_and_save_histogram(data, title):
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

    plt.savefig(f'{title}.png')

    plt.close()


def load_and_threshold_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return bin_img


def find_segments(profile):
    start = None
    segments = []
    for i, val in enumerate(profile):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(profile)))
    return segments


def segment_characters(bin_img):
    profile_x = np.sum(bin_img, axis=0)
    profile_y = np.sum(bin_img, axis=1)

    plot_and_save_histogram(profile_x, "output/x_hist")
    plot_and_save_histogram(profile_y, "output/y_hist")

    maps = {}
    max_key = -1
    max_value = 0
    for i in range(len(profile_x)):
        if profile_x[i] in maps:
            maps[profile_x[i]] += 1
        else:
            maps[profile_x[i]] = 1
        if maps[profile_x[i]] > max_value:
            max_value = maps[profile_x[i]]
            max_key = profile_x[i]

    for i in range(len(profile_x)):
        balance = (max_key + 300)
        if profile_x[i] - balance >= 0:
            profile_x[i] -= balance
        else:
            profile_x[i] = 0

    segments_x = find_segments(profile_x)
    segments_y = find_segments(profile_y)

    rectangles = []
    for sx in segments_x:
        for sy in segments_y:
            rectangles.append((sx[0], sy[0], sx[1], sy[1]))
    return rectangles


def draw_rectangles(img, rectangles):
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)


def extract_and_save_segments(bin_img, rectangles, base_path='output/characters/character'):
    for idx, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        segment = bin_img[y1:y2, x1:x2]
        cv2.imwrite(f'{base_path}_{idx}.png', segment)


def process_images(image_path):
    bin_img = load_and_threshold_image(image_path)
    rectangles = segment_characters(bin_img)

    inverted_img = 255 - bin_img
    colored_img = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)
    draw_rectangles(colored_img, rectangles)

    cv2.imwrite('output/segmented_characters.png', colored_img)

    extract_and_save_segments(bin_img, rectangles)


if __name__ == "__main__":
    image_path = "input/Phraze.bmp"
    process_images(image_path)

    generate_markdown_report(
        image_path="output/segmented_characters.png",
        segments_path="output/characters/",
        histogram_paths=("output/x_hist.png", "output/y_hist.png")
    )