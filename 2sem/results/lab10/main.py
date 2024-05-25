from collections import defaultdict

from scipy import interpolate
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def integral_image(img: np.array) -> np.array:
    cumsum_img = img.copy()

    cumsum_img = np.cumsum(cumsum_img, axis=0)

    cumsum_img = np.cumsum(cumsum_img, axis=1)
    return cumsum_img


def sum_in_frame(integral_img: np.array, x: int, y: int, frame_size: int):
    len = integral_img.shape[1] - 1
    hight = integral_img.shape[0] - 1

    half_frame = frame_size // 2
    above = y - half_frame - 1
    low = y + half_frame
    left = x - half_frame - 1
    right = x + half_frame

    A = integral_img[max(above, 0), max(left, 0)]
    B = integral_img[max(0, above), min(len, right)]
    C = integral_img[min(hight, low), max(left, 0)]
    D = integral_img[min(hight, low), min(right, len)]

    if max(left + 1, 0) == 0 and max(above + 1, 0) == 0:
        return D
    elif max(left + 1, 0) == 0:
        return D - B
    elif max(above + 1, 0) == 0:
        return D - C

    return D - C - B + A


def culculate_mean(integral_image: np.array, x: int, y: int, frame_size):
    square = frame_size ** 2
    s = sum_in_frame(integral_image, x, y, frame_size)
    return s // square


def change_sample_rate(path, new_sample_rate=22050):
    audioPath = "input/" + path
    old_rate, old_audio = wavfile.read(audioPath)

    if old_rate != new_sample_rate:
        # Вычисление продолжительности аудиозаписи в секундах
        duration = old_audio.shape[0] / old_rate

        time_old = np.linspace(0, duration, old_audio.shape[0])
        time_new = np.linspace(0, duration, int(old_audio.shape[0] * new_sample_rate / old_rate))

        # Линейная интерполяция аудиоданных на новой временной сетке
        interpolator = interpolate.interp1d(time_old, old_audio.T)
        new_audio = interpolator(time_new).T

        wavfile.write("results/wavs/" + path, new_sample_rate, np.round(new_audio).astype(old_audio.dtype))


def find_formants(freqs, integral_spec, x, frame_size):
    res = [0] * integral_spec.shape[0]

    for i in range(1, integral_spec.shape[0], frame_size):
        res[i] = culculate_mean(integral_spec, x, i, frame_size)

    origin = res.copy()
    res.sort(reverse=True)

    res = res[:3]

    return list(map(lambda power: (int(freqs[origin.index(power)]), int(power)), res))


def find_all_formants(freqs, integral_spec, frame_size):
    res = set()
    for i in range(integral_spec.shape[1]):
        formant = find_formants(freqs, integral_spec, i, frame_size)
        form = list(map(lambda bind: bind[0], formant))
        for j in range(3):
            res.add(form[j])

    res.discard(0)
    return res


def power(freqs, integral_spec, frame_size, formant_s):
    power_dict = defaultdict(float)

    for i in range(integral_spec.shape[1]):
        for j in find_formants(freqs, integral_spec, i, frame_size):
            if j[0] != 0:
                power_dict[j[0]] += j[1]

    return {k: power_dict[k] for k in formant_s}


def spectrogram_plot(sample, sample_rate, t=11000):
    frequencies, times, my_spectrogram = signal.spectrogram(sample, sample_rate, scaling='spectrum', window=('hann'))
    spec = np.log10(my_spectrogram)
    plt.pcolormesh(times, frequencies, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())

    plt.ylim(top=t)
    plt.yticks(np.arange(min(frequencies), max(frequencies), 500))
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')
    return my_spectrogram, frequencies


def process_wav_file(filename, save_path, formant_lines=None):
    change_sample_rate(filename)
    sample_rate, sample = wavfile.read(f"results/wavs/{filename}")
    spectogram, frequencies = spectrogram_plot(sample, sample_rate, 11000)

    if formant_lines:
        for idx, y in enumerate(formant_lines):
            label = "Форманты" if idx == 0 else None
            plt.axhline(y=y, color='r', linestyle='-', lw=0.5, label=label)
        plt.legend()

    plt.savefig(save_path, dpi=500)
    plt.clf()

    return spectogram, frequencies


def analyze_spectrogram(spectrogram, frequencies, frame_size, label,  need_formant=True):
    integral = integral_image(spectrogram)
    formants = list(find_all_formants(frequencies, integral, frame_size))
    formants.sort()

    print(f"\nМинимальная частота для звука {label}: {formants[0]}")
    print(f"Максимальная частота для звука {label}: {formants[-1]}")
    if need_formant:
        print(f"Тембрально окрашенный тон для звука {label}: {formants[0]}")
        power_values = power(frequencies, integral, frame_size, formants)
        print(f"Три самые сильные форманты: {sorted(power_values, key=lambda i: power_values[i], reverse=True)[:3]}\n")


if __name__ == '__main__':
    spectrogram_a, frequencies_a = process_wav_file("a.wav", "results/spectrogram_a.png", [86, 602, 861])
    analyze_spectrogram(spectrogram_a, frequencies_a, 3, "А")

    spectrogram_i, frequencies_i = process_wav_file("i.wav", "results/spectrogram_i.png", [86, 344, 602])
    analyze_spectrogram(spectrogram_i, frequencies_i, 3, "И")

    spectrogram_gav, frequencies_gav = process_wav_file("gav.wav", "results/spectrogram_gav.png")
    analyze_spectrogram(spectrogram_gav, frequencies_gav, 5, "ГАВ", need_formant=False)