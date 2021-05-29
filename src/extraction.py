from scipy import signal
import numpy as np
import librosa
import glob
import pandas as pd


def get_signal(path, fs=10000):
    signal_fm, sr = librosa.load(path, mono=True, sr=fs)
    signal_fm = signal_fm.astype('float32')
    return signal_fm, sr


def draw_spectogram(ax, fig, signal_fm, title):
    D = librosa.stft(signal_fm)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax, cmap='binary_r')
    ax.set(title=title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return S_db


def get_base_freq(signal_fm, fs):
    corr = signal.correlate(signal_fm, signal_fm, mode='full')
    corr = corr[int(corr.size / 2):]
    peaks, _ = signal.find_peaks(corr, height=0)
    res = np.where(corr == (np.max(corr[peaks])))
    return fs / res[0][0]


def get_formants(signal_fm, fs):
    N = len(signal_fm)
    w = np.hamming(N)

    # Okno i filtr górnoprzepustowy.
    s1 = signal_fm * w
    s1 = signal.lfilter([1], [1., 0.63], s1)

    # Filtr LPC.
    ncoeff = 2 + fs / 1000
    A = librosa.core.lpc(s1, int(ncoeff))

    # Pierwiastki.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Kąty.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Częstotliwości formantów.
    return sorted(angz * (fs / (2 * 3.14)))


def create_database(path_data="C:/Users/kb39309/Documents/Studia/Vovel_sounds/src/vowels_dataset/*", path_csv="data.csv"):
    filenames = glob.glob(path_data)

    df = pd.DataFrame(columns=["file", "f0", "form1", "form2", "form3", "form4", "sex"])

    for fname in filenames:

        input, fs = get_signal(fname, fs=10000)

        f0 = get_base_freq(input, fs)
        formants = get_formants(input, fs)

        df = df.append({"file": fname,
                        "f0": f0,
                        "form1": formants[0],
                        "form2": formants[1],
                        "form3": formants[2],
                        "form4": formants[3],
                        "sex": 1 if 'Ola' in fname else 0}, ignore_index=True)
    df.to_csv(path_csv)

