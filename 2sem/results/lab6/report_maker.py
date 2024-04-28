import os


def generate_markdown_report(image_path, segments_path, histogram_paths):
    """
    Generates a markdown report for the image processing results.

    :param image_path: Path to the segmented characters image.
    :param segments_path: Path to the directory containing the extracted character segments.
    :param histogram_paths: Tuple or list containing paths to the x and y histograms.
    """
    report_content = """
# Отчет по обработке изображения

## Введение

Процесс анализа и сегментации символов на изображении подразумевает следующие шаги: загрузка изображения, его бинаризация, анализ и визуализация гистограмм, выделение сегментов символов, их сохранение и визуализация.

## Гистограммы

### Горизонтальная гистограмма

![Горизонтальная гистограмма]({x_hist})

### Вертикальная гистограмма

![Вертикальная гистограмма]({y_hist})

## Изображение с выделенными символами

![Изображение с сегментированными символами]({segmented_image})

## Сохраненные сегменты символов

""".format(
        x_hist=histogram_paths[0],
        y_hist=histogram_paths[1],
        segmented_image=image_path
    )

    segment_files = os.listdir(segments_path)
    for filename in segment_files:
        img_path = os.path.join(segments_path, filename)
        report_content += "- ![]({})\n".format(img_path)

    with open("image_processing_report.md", "w") as file:
        file.write(report_content)
