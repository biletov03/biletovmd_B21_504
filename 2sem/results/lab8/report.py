def generate_md_report(image_names, base_features, contrasted_features, output_path='report.md'):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('# Отчет о характеристиках текстур изображений\n\n')

        for name in image_names:
            base_img = f'input/{name}'
            semi_img_path = f'results/semitone/{name}'
            contrasted_img_path = f'results/contrasted/{name}'
            histogram_path = f'results/histograms/{name}'
            haralick_path = f'results/haralik/{name}'
            haralick_contrasted_path = f'results/haralik_contrasted/{name}'

            file.write(f'## {name}\n')
            file.write('### Исходное изображение\n')
            file.write(f'![Исходное изображение]({base_img})\n\n')
            file.write('### Полутоновое изображение\n')
            file.write(f'![Исходное изображение]({semi_img_path})\n\n')
            file.write('### Контрастированное изображение\n')
            file.write(f'![Контрастированное изображение]({contrasted_img_path})\n\n')
            file.write(f'![Гистограмма]({histogram_path})\n\n')
            file.write('### Характеристики Харалика для исходного изображения\n')
            file.write(f'```\n{base_features[name]}\n```\n\n')
            file.write('### Характеристики Харалика для контрастированного изображения\n')
            file.write(f'```\n{contrasted_features[name]}\n```\n\n')
            file.write('### Матрица Харалика для исходного изображения\n')
            file.write(f'![Матрица Харалика для исходного изображения]({haralick_path})\n\n')
            file.write('### Матрица Харалика для контрастированного изображения\n')
            file.write(f'![Матрица Харалика для контрастированного изображения]({haralick_contrasted_path})\n\n')
