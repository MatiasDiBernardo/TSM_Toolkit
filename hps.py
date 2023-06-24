import numpy as np
from scipy.signal import medfilt as mediann
from librosa import stft, istft

def hps(x, N, M):
    """ Divides the percusive and harmonics components from the input signal
    returning two diferent signals.

    Args:
        x (np.array): Audio signal.
        fs (int): Sample rate.
        N (int): Window lenght for STFT.
        M (int): Median filter window leight, this value must be 
        odd. In the contrary it will be forced to the upper inmedate.

    Returns:
        xp (np.array): Percusive signal.
        xh (np.array): Harmonic signal.
    """
    #Obtain the STFT and the spectogram Y 
    X = stft(x, n_fft = N )
    Y = abs(X)

    Yh = mediann(Y,(1,M)) #2D median filter with window
    Yp = mediann(Y,(M,1))   

    Mh = np.zeros(np.shape(Yh))
    Mp = np.zeros(np.shape(Yh))

    a , b = np.shape(Yh)
    # Evaluamos para cada frecuencia y segmento 
    for i in range(a):
        for l in range(b):
            if Yh[i,l] > Yp[i,l]:
                Mh[i,l] = 1
            else:
                Mp[i,l] = 1 

    Xp = X * Mp
    Xh = X * Mh

    #Inverse transform for each isolated spectrum 
    xp = istft(Xp, n_fft = N)
    xh = istft(Xh, n_fft = N)

    #Compensate the gap created by the median filter
    l = len(x)-len(xp)
    xp = np.concatenate((xp, np.zeros(l)))
    xh = np.concatenate((xh, np.zeros(l)))
  
    return xp, xh
