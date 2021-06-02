from scipy import signal
import numpy as np
import librosa
import glob
import pandas as pd
import os


def get_signal(path, fs=10000):
    """ Funkcja wczytująca sygnał zmonofonizowany z zadanej ścieżki.

    :param path: Ścieżka w postaci napisu
    :param fs: Częstotliwość w której chcemy odczytać sygnał
    :return: Sygnał zmonofonizowany oraz częstotliwość (Tupla)
    """
    signal_fm, sr = librosa.load(path, mono=True, sr=fs)
    signal_fm = signal_fm.astype('float32')
    return signal_fm, sr


def get_envelope(signal_r, range_value=5000):
    """ Funkcja obliczająca obwiednię dla zadanego sygnału.

    :param signal_r: Sygnał w postaci zmonofonizowanej (jeden kanał)
    :param range_value: Krok w obwiedni
    :return: Indeksy obwiedni górnej oraz dolnej, a także wartości obwiedni
    """
    max_values = []
    min_values = []
    ind_max = []
    ind_min = []

    for i in range(0, len(signal_r), range_value):
        inx_max = np.argmax(signal_r[i: i + range_value])
        ind_max.append(inx_max + i)

        inx_min = np.argmin(signal_r[i: i + range_value])
        ind_min.append(inx_min + i)

        max_values.append(np.max(signal_r[i: i + range_value]))
        min_values.append(np.min(signal_r[i: i + range_value]))

    return ind_max, ind_min, max_values, min_values


def draw_spectrogram(ax, fig, signal_fm, title):
    """ Funkcja tworząca wykres spektrogramu.

    :param ax: Okno figury (matplotlib.axis)
    :param fig: Figura (matplotlib.figure)
    :param signal_fm: Sygnał w postaci zmonofonizowanej (jeden kanał)
    :param title: Podpis okna (matplotlib.title)
    :return: Sygnał [dB]
    """
    D = librosa.stft(signal_fm)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax, cmap='binary_r')
    ax.set(title=title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return S_db


def get_base_freq(signal_fm, fs):
    """ Funkcja wydobywająca bazową częstotliwość z sygnału podanego jako argument.

    :param signal_fm: Sygnał w postaci zmonofonizowanej (jeden kanał)
    :param fs: Częstotliwość sygnału
    :return: Częstotliwość podstawowa
    """
    corr = signal.correlate(signal_fm, signal_fm, mode='full')
    corr = corr[int(corr.size / 2):]
    peaks, _ = signal.find_peaks(corr, height=0)
    res = np.where(corr == (np.max(corr[peaks])))
    return fs / res[0][0]


def get_formants(signal_fm, fs):
    """ Funkcja uzyskująca formanty z sygnału.

    :param signal_fm: Sygnał w postaci zmonofonizowanej (jeden kanał)
    :param fs: Częstotliwość sygnału
    :return: Formanty (Tupla)
    """
    N = len(signal_fm)
    w = np.hamming(N)
    s1 = signal_fm * w
    s1 = signal.lfilter([1], [1., 0.63], s1)
    ncoeff = 2 + fs / 1000
    A = librosa.core.lpc(s1, int(ncoeff))
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    return sorted(angz * (fs / (2 * 3.14)))


def create_database(path_data="C:/Users/kb39309/Documents/Studia/Vovel_sounds/src/vowels_dataset/*", path_csv="data.csv"):
    """ Funkcja pozyskująca cechy z próbek dźwiękowych do ramki danych (Pandas).

    :param path_data: Ścieżka do plików dźwiękowych (glob)
    :param path_csv: Ścieżka do zapisu ramki danych z rozszerzeniem .csv
    :return: Ramka danych (pandas.DataFrame)
    """
    filenames = glob.glob(path_data)

    df = pd.DataFrame(columns=["vowel", "sex", "f0", "form1", "form2", "form3", "form4"])

    for fname in filenames:

        input, fs = get_signal(fname, fs=10000)

        f0 = get_base_freq(input, fs)
        formants = get_formants(input, fs)

        df = df.append({'vowel': os.path.basename(fname)[2],
                        "sex": 1 if 'f' in fname else 0,
                        "f0": f0,
                        "form1": formants[0],
                        "form2": formants[1],
                        "form3": formants[2],
                        "form4": formants[3]}, ignore_index=True)
    df.to_csv(path_csv)
    return df

