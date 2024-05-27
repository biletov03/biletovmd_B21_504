import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def find_segments(profile, threshold=0):
    start = None
    segments = []
    for i, val in enumerate(profile):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(profile)))
    return segments


def segment_characters(bin_img):
    profile_x = np.sum(bin_img, axis=0)
    profile_y = np.sum(bin_img, axis=1)

    segments_x = find_segments(profile_x, threshold=0)
    rectangles = []

    for sx in segments_x:
        segment_slice = bin_img[:, sx[0]:sx[1]]
        profile_y_segment = np.sum(segment_slice, axis=1)
        segments_y = find_segments(profile_y_segment, threshold=0)
        for sy in segments_y:
            rectangles.append((sx[0], sy[0], sx[1], sy[1]))
    return rectangles

def draw_rectangles(img, rectangles):
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)


def extract_and_save_segments(bin_img, rectangles, base_path='output/characters/character/'):
    for idx, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        segment = bin_img[y1:y2, x1:x2]
        inverted_segment = 255 - segment
        cv2.imwrite(f'{base_path}{idx}.png', inverted_segment)


def process_images(image_path):
    prefix = "input/strings/"
    bin_img = load_and_threshold_image(prefix + image_path + ".bmp")
    rectangles = segment_characters(bin_img)

    inverted_img = 255 - bin_img
    colored_img = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)
    draw_rectangles(colored_img, rectangles)

    cv2.imwrite(f'output/{image_path}segmented_characters.png', colored_img)

    extract_and_save_segments(bin_img, rectangles, f"input/{image_path}/")


if __name__ == "__main__":
    image_paths = ["base", "wide"]
    for path in image_paths:
        process_images(path)
