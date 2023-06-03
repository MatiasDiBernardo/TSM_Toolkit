from scipy import signal
import numpy as np

"""
Esta parte del script por ahí se puede sacar porque es solo llamar a la función de scipy
si por ahí necesitamos alguna ventana que o esta en signal ahí si pero creo que estan todas.
"""

def window_normalization(w, Hs):
    """Calculates the normalization factor
    according to the current window.

    Args:
        w (np.array): Window used in the OLA process.
        Hs (int): Synthesis hopsize.

    Returns:
        np.array: Normalization signal.
    """
    N = len(w)
    normalization_signal = np.copy(w)
    for n in range(1, int(N/Hs)):
        w_desp = np.roll(w, int(n*Hs))
        normalization_signal += w_desp
    
    return normalization_signal
    

def get_window(N, type):
    """Generate different types of windows.

    Args:
        N (int): Long of the windows in samples.
        type (string): Type of window, ex. Hann, Blackman, Hamming.
    """
    return None