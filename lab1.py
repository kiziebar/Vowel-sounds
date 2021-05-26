from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import copy
import matplotlib
import matplotlib.gridspec as gridspec

import sys
import numpy
import wave
import math
from scipy.signal import lfilter, hamming
import librosa
import glob
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

#dostepne pliki w bazie
filenames = glob.glob("F:/S2_Semestr3/sygnaly_projekt/vowels_dataset/*")
#print(filenames)

def monofonization(data):
    sygnal_mono = np.mean(data, axis = 1)
    return sygnal_mono

Dane = {'Plik' : [],
        'f0' : [],
        'formant1' : [],
        'formant2' : [],
        'formant3' : [],
        'formant4' : [],
        'formant5' : [],
        'formant6' : [],
        'plec' : [],
        }

for fname in filenames:
    input, fs = sf.read(fname)
    if len(np.shape(input)) > 1:
        pass
    else:
        input = input.reshape(-1,1)
    
    input = monofonization(input)
    
    corr = signal.correlate(input, input, mode='full')
    corr = corr[int(corr.size / 2):]
    peaks, _ = signal.find_peaks(corr, height=0)
    res = np.where(corr == (np.max(corr[peaks])))
    #print(f'Dominant frequency: {fs / res[0][0]} Hz\n\n')
    Dane['Plik'].append(fname)
    Dane['f0'].append(fs / res[0][0])

    x = copy.deepcopy(input)

    #pierwszy sposob wyznaczania formantow
    # Get Hamming window.
    N = len(x)
    w = np.hamming(N)
    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1., -0.63], 1, x1)
    # Get LPC.
    A = librosa.core.lpc(x1, 8)
    # Get roots.
    rts = np.roots(A)
    #rts = [r for r in rts if np.imag(r) >= 0]
    rts = rts[np.imag(rts) >= 0]
    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))
    # Get frequencies.
    #frqs = sorted(angz * (fs / (2 * np.pi)))
    frqs = angz * fs / (2 * np.pi)
    ind = np.argsort(frqs)
    frqs = frqs[ind]
    print(frqs)
    


    '''
    #Drugi sposob wyznaczania formantow
    A = librosa.core.lpc(x, 8)
    rts = np.roots(A)
    rts = rts[np.imag(rts) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    frqs = angz * fs / (2 * np.pi)
    ind = np.argsort(frqs)
    frqs.sort()
    #bw = -1 / 2 * (fs / (2 * 3.14)) * np.log(abs(rts[ind]));
    print(frqs)
    frqs = frqs[frqs > 0]
    print(frqs[0])
    '''

    
    Dane['formant1'].append(frqs[0])
    Dane['formant2'].append(frqs[1])
    Dane['formant3'].append(frqs[2])
    Dane['formant4'].append(frqs[3])
    if len(frqs) > 4:
        Dane['formant5'].append(frqs[4])
    else:
        Dane['formant5'].append(0)
    if len(frqs) > 5:
        Dane['formant6'].append(frqs[5])
    else:
        Dane['formant6'].append(0)
    if 'Ola' in fname:
        Dane['plec'].append(1)
    else:
        Dane['plec'].append(0)
    
    
data = pd.DataFrame(Dane)
data.to_csv('created_dataset.csv')