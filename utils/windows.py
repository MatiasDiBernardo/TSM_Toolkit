import numpy as np

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
    