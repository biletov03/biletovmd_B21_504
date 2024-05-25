from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import path
from functions import calculate_glcm, calculate_haralick_features_from_glcm, gamma_correction, to_semitone
from report import generate_md_report


if __name__ == '__main__':
    image_paths = {"table.jpg": 2, "keyboard.png": 0.5, "wall.png": 1.2}
    base_features = {}
    contrasted_features = {}
    paths = []
    for image_path in image_paths.keys():
        semitone_img = to_semitone("input/" + image_path)
        semitone_img.save(path.join('results','semitone', image_path))

        semi = np.array(Image.open(path.join('results', 'semitone', image_path)).convert('L'))

        transformed = gamma_correction(semi, image_paths[image_path])
        transformed_img = Image.fromarray(transformed.astype(np.uint8), "L")
        transformed_img.save(path.join('results', 'contrasted', image_path))

        figure, axis = plt.subplots(2, 1)
        axis[0].hist(x=semi.flatten(), bins=np.arange(0, 255))
        axis[0].title.set_text('Исходное изображение')

        axis[1].hist(x=transformed.flatten(), bins=np.arange(0, 255))
        axis[1].title.set_text('Преобразованное изображение')
        plt.tight_layout()
        plt.savefig(path.join('results', 'histograms', image_path))

        matrix = calculate_glcm(semi.astype(np.uint8))
        result = Image.fromarray(matrix.astype(np.uint8), "L")
        result.save(path.join('results', 'haralik', image_path))

        t_matrix = calculate_glcm(transformed.astype(np.uint8))
        t_result = Image.fromarray(t_matrix.astype(np.uint8), "L")
        t_result.save(path.join('results', 'haralik_contrasted', image_path))

        base_features[image_path] = calculate_haralick_features_from_glcm(matrix)

        contrasted_features[image_path] = calculate_haralick_features_from_glcm(t_matrix)
        paths.append(image_path)

    generate_md_report(paths, base_features, contrasted_features)
