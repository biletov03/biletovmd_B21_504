import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import librosa.display


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def save_spectrogram(y, sr, title, output_dir, filename):
    window_length = 2048  # Задаем длину окна
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=window_length, window=np.hanning(window_length))),
                                ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()


def find_min_max_frequencies(y, sr, fmin=50, fmax=4000):
    window_length = 2048  # Задаем длину окна
    S = np.abs(librosa.stft(y, n_fft=window_length, window=np.hanning(window_length)))
    freqs = librosa.fft_frequencies(sr=sr)

    valid_mask = (freqs >= fmin) & (freqs <= fmax)
    S_filtered = S[valid_mask, :]
    freqs_filtered = freqs[valid_mask]

    mean_amplitudes = np.mean(S_filtered, axis=1)

    peaks, _ = find_peaks(mean_amplitudes, height=np.mean(mean_amplitudes))

    if peaks.size > 0:
        min_freq = freqs_filtered[peaks].min()
        max_freq = freqs_filtered[peaks].max()
    else:
        min_freq = fmin
        max_freq = fmax

    return min_freq, max_freq


def find_fundamental_harmonics(y, sr):
    y_harmonic, _ = librosa.effects.hpss(y)
    pitches, magnitudes = librosa.core.piptrack(y=y_harmonic, sr=sr)
    harmonics = []
    for t in range(magnitudes.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            harmonics.append(pitch)
    if len(harmonics) > 0:
        return np.bincount(harmonics).argmax()
    return None



def find_formants(y, sr, num_formants=3):
    window_length = 2048
    D = np.abs(librosa.stft(y, n_fft=window_length, window=np.hanning(window_length)))
    freqs = librosa.fft_frequencies(sr=sr)
    formants = []

    for t in range(D.shape[1]):
        peak_indices, _ = find_peaks(D[:, t], distance=sr / (50 if sr > 1000 else 1))
        if len(peak_indices) >= num_formants:
            peak_freqs = freqs[peak_indices]
            peak_amps = D[peak_indices, t]
            sorted_indices = np.argsort(peak_amps)[-num_formants:]
            formants_at_time = peak_freqs[sorted_indices]
            formants.append(formants_at_time)

    formants = np.array(formants)
    avg_formants = np.mean(formants, axis=0)
    return avg_formants


def main():
    input_files = ['input/a.wav', 'input/i.wav', 'input/gav.wav']
    output_dir = 'results'
    for file in input_files:
        print(f'Analyzing {file}')
        y, sr = load_audio(file)

        filename = os.path.basename(file).replace('.wav', '_spectrogram.png')
        save_spectrogram(y, sr, f'Spectrogram of {file}', output_dir, filename)

        min_freq, max_freq = find_min_max_frequencies(y, sr)
        print(f'Min frequency for {file}: {min_freq} Hz')
        print(f'Max frequency for {file}: {max_freq} Hz')

        fundamental_harmonics = find_fundamental_harmonics(y, sr)
        if fundamental_harmonics:
            print(f'Fundamental harmonics for {file}: {fundamental_harmonics} Hz')
        else:
            print(f'Could not find fundamental harmonics for {file}')

        formants = find_formants(y, sr)
        print(f'Top 3 formants for {file}: {formants} Hz\n')


if __name__ == '__main__':
    main()
