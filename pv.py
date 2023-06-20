import numpy as np
#from librosa import stft, istft
from pytsmod import stft, istft

def TSM_PV(x, fs, N, alpha, Hs):
    """Alpies TSM procedure base on phase vocoder.

    Args:
        x (np.array): Audio signal.
        fs (int): Sample rate.
        N (int): Window lenght.
        alpha (float): Stretching factor.
        Hs (int): Synthesis hopsize length.

    Returns:
        np.array: Time modify audio signal.
    """

    #Define parameters
    Ha = int(Hs/alpha)  #Hop analysis
    delta_t = Ha/fs
    win_type = "hann"
    #X = stft(x, n_fft=N, hop_length=Ha, win_length=N, window=win_type)
    X = stft(x, Ha, win_type, N, sr=fs)
    Y = np.zeros(np.shape(X), dtype=complex)  #Output STFT
    Y[:, 0] = X[:, 0]  # phase initialization

    k = np.arange(N / 2 + 1)  #Stft uses only postive side
    omega = k * fs/N  #From 0 to fs/2

    for i in range(1, X.shape[1]):
        #Save current and last mod phase
        phi_curr = (np.angle(X[:, i])) / (2*np.pi)  #Phase between (-0.5, 0.5)
        phi_last = (np.angle(Y[:, i - 1]))/ (2*np.pi)

        #Calculates ideal next phase
        phi_pred = phi_last + omega * delta_t
        
        #Diference between phases and normalization
        phi_difference = phi_curr - phi_pred - (i*Hs)/N 
        normalize_phase = np.mod(phi_difference + 0.5, 1) - 0.5  #Mod bewteen (-0.5, 0.5)
        phi_error = (Hs/N) * normalize_phase  #Bin offset

        #Calculate Instantanious frequency
        IF_w = (i + phi_error) * fs / N
        
        #Phase modification using IF
        phi_mod = np.angle(Y[:, i - 1]) + (IF_w * Hs/fs)

        #Add modify frame to output stft
        Y[:, i] = np.abs(X[:, i]) * np.exp(2*np.pi * 1j * phi_mod) 


    #y = istft(Y, hop_length=Hs, win_length=N, n_fft=N, window=win_type)
    y = istft(Y, Hs, win_type, N)
    
    return y