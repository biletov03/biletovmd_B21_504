from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def spectrogram_plot(samples, sample_rate, title='Spectrogram', ylim=20000, use_log_scale=True):
    plt.figure(figsize=(10, 4))
    frequencies, times, my_spectrogram = signal.spectrogram(samples, sample_rate, scaling='spectrum', window='hann')
    spec = np.log10(my_spectrogram + 1e-10)  # Логарифмируем и добавляем небольшую константу, чтобы избежать log(0)

    plt.pcolormesh(times, frequencies, spec, shading='gouraud')
    if use_log_scale: plt.yscale('log')

    plt.colorbar(label='Логарифм спектральной плотности')
    plt.ylim(top=ylim, bottom=frequencies[1])  # Исключаем нулевую частоту из log scale
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')
    plt.title(title)


def denoise(samples, sample_rate, method='butter', **kwargs):
    if method == 'butter':
        b, a = signal.butter(3, kwargs['cutoff'] / (0.5 * sample_rate), btype='low')
        return signal.filtfilt(b, a, samples)
    elif method == 'savgol':
        return signal.savgol_filter(samples, **kwargs)
    else:
        raise ValueError("Неизвестный метод фильтрации")


def detect_high_energy_moments(spec, time_resolution, delta_t=0.1):
    time_window = int(delta_t / time_resolution)

    moments = []

    for t in range(spec.shape[1] - time_window):
        energy = np.sum(np.absolute(spec[:, t:t + time_window]) ** 2)

        moments.append((t * time_resolution, energy))

    max_energy_moment = max(moments, key=lambda x: x[1])[0]

    return max_energy_moment


def to_pcm(y):
    return np.int16(y / np.max(np.abs(y)) * 32767)


if __name__ == '__main__':
    dpi = 500
    sample_rate, samples = wavfile.read('input/Гитара.wav')
    cutoff_frequency = 3000

    # 1. Спектрограмма с логарифмической шкалой частот и сохранение
    spectrogram_plot(samples, sample_rate, title='Исходная спектрограмма', use_log_scale=True)
    plt.savefig('output/original_spectrogram_log_scale.png', dpi=dpi)
    plt.clf()

    # 2. Денойзинг с фильтром Баттерворта
    denoised_butter = denoise(samples.astype(float), sample_rate, method='butter', cutoff=cutoff_frequency)
    spectrogram_plot(denoised_butter, sample_rate, title='Денойзинг с фильтром Баттерворта', use_log_scale=True)
    plt.savefig('output/denoised_spectrogram_butter.png', dpi=dpi)
    plt.clf()
    wavfile.write('output/denoised_butter.wav', sample_rate, to_pcm(denoised_butter))

    # 3. Денойзинг с фильтром Савицкого-Голея
    denoised_savgol = denoise(samples.astype(float), sample_rate, method='savgol', window_length=101, polyorder=3)
    spectrogram_plot(denoised_savgol, sample_rate, title='Денойзинг с фильтром Савицкого-Голея', use_log_scale=True)
    plt.savefig('output/denoised_spectrogram_savgol.png', dpi=dpi)
    plt.clf()
    wavfile.write('output/denoised_savgol.wav', sample_rate, to_pcm(denoised_savgol))

    # 4. Обнаружение моментов с наибольшей энергией
    frequencies, times, spectrogram = signal.spectrogram(denoised_butter, sample_rate)
    time_resolution = times[1] - times[0]

    high_energy_moment = detect_high_energy_moments(spectrogram, time_resolution)

    print(f"Момент времени с наибольшей энергией: {high_energy_moment} с")
