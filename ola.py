#Implementacion OLA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from utils.wav_utils import read_wav, save_wav
from utils.windows import window_normalization

def TSM_OLA(x, N, alpha, Hs): 
    """Applies TSM procedure based on OLA: Overlapp and Add.

    Args:
        x (np.array): Audio signal.
        N (int): Window lenght.
        alpha (float): Stretching factor.
        Hs (int): Synthesis hopsize length.

    Returns:
        np.array: Time modify audio signal.
    """

    #defining output vector size according to scale factor
    if alpha == 1:
        return x 
    else:
        y = np.zeros(int(len(x) * alpha) + N)  #N accounts for last frame.
     
    #Window election
    win_type = "hann"
    w = get_window(win_type, N)
    w_norm = window_normalization(w, Hs) 

    #Inicialization
    Ha = int(Hs/alpha)
    cicles = int((len(x) - N)/Ha) + 1  #Amount of frames in the signal.

    for m in range(cicles):

        #segment input into analysis frames 
        x_m = x[m * Ha: N + (m * Ha)]

        #compute output signal by windowing x_m and normalizing
        #and overlapping according to Hs
        y[m * Hs: N + (m * Hs)] += (x_m*w)/(w_norm)
        #y[m * Hs: N + (m * Hs)] += (x_m * w[:len(x_m)]) / w_norm[:len(x_m)] #chatGPT
                
    return y 


