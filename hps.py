#Aca va la implementación del HPS
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import stft 
from scipy.signal import get_window
from scipy.ndimage import median_filter as median
from librosa import stft, istft

from utils.wav_utils import read_wav, save_wav
from utils.windows import window_normalization

#parte de prueba temporal para definir señales
N = 10000
fs = 44100
n = np.arange(0, N, 1) 

def TSM(x, fs, N):
    """ Divides the percusive and harmonics components from the input signal
    returning two diferent signals.

    Args:
        x (np.array): Audio signal.
        fs (int): Sample rate.
        N (int): Window lenght for FFT.

    Returns:
        xp (np.array): Percusive signal.
        xh (np.array): Harmonic signal.
    """
    #Obtain the STFT and the spectogram Y 
    X = stft(x, fs, window= 'hann', nfft = N)
    Y = abs( )

    #Calculate horizontal and vertical median filter of X
    Y_p = median(X)

    
    return() 

# %%
