import numpy as np
import librosa

#Ver tema factor N y arregar el tema de importar cosas de otras folders.

def phase_vocoder(x, fs, N, alpha, Hs):

    if alpha == 1:
        return x
    if alpha < 1:
        y = np.zeros(int(len(x) * alpha) + N)  #N es factor de compensasión pero se puede sacar
    if alpha > 1:
        y = np.zeros(int(len(x) * alpha))  #N es factor de compensasión pero se puede sacar
    
    Ha = int(Hs/alpha)
    
    cicles = int((len(x) - N)/Ha) + 1
    
    for i in range(cicles):
        Xm = x[i * Ha: N + i * Ha]
        y[i * Hs: N + i * Hs] += Xm
        
    return y

from utils.wav_utils import read_wav, save_wav


fs = 22050
x, _ = read_wav("data\piano_cerca.wav", fs)

rta = phase_vocoder(x, fs, 2048, 1.2, 512)

save_wav(rta, fs, "data\\test_ola.wav")