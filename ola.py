#Implementacion OLA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from utils.wav_utils import read_wav, save_wav
from utils.windows import window_normalization

def TSM_OLA(x, fs, N, alpha, Hs):
    """Applies TSM procedure based on OLA: Overlapp and Add.

    Args:
        x (np.array): Audio signal.
        fs (int): Sample rate.
        N (int): Window lenght.
        alpha (float): Stretching factor.
        Hs (int): Synthesis hopsize length.

    Returns:
        np.array: Time modify audio signal.
    """

    if alpha == 1:
        return x
    if alpha < 1:
        y = np.zeros(int(len(x) * alpha) + N)  #N accounts for last frame.
    if alpha > 1:
        y = np.zeros(int(len(x) * alpha)) 
    
    #Window election
    win_type = "hann"
    w = get_window(win_type, N)
    w_norm = window_normalization(w, Hs)

    #Inicialization
    Ha = int(Hs/alpha)
    cicles = int((len(x) - N)/Ha) + 1  #Amount of frames in the signal.

    for m in range(cicles):

        #compute analysis frames
        x_m = x[m * Ha: N + (m * Ha)]
       
        
    return 
