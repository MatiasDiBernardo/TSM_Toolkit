import numpy as np
from librosa import stft, istft

def normalize_phase(phi_diffrenece):
    """Normalice the value of difference between
    the real and the predicted phase.

    Args:
        phi_diffrenece (np.array): Phase difference.
    Returns:
        (np.array): Normalize phase array.
    """
    normalize_phase = np.zeros(len(phi_diffrenece))
    max_phase = 0.5
    
    for i, phi in enumerate(phi_diffrenece):
        if phi > max_phase:
            normalize_phase[i] = phi % max_phase
            continue
        if phi < -max_phase:
            normalize_phase[i] = -1 * ((-1*phi)%max_phase)
            continue
        normalize_phase[i] = phi

    return normalize_phase 

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
    X = stft(x, n_fft=N, hop_length=Ha, win_length=N, window=win_type)
    Y = np.zeros(np.shape(X), dtype=complex)  #Output STFT
    Y[:, 0] = X[:, 0]  # phase initialization

    k = np.arange(N / 2 + 1)  #Stft uses only postive side
    omega = k * fs/N  #From 0 to fs/2

    for i in range(1, X.shape[1]):
        #Save current and last mod phase
        phi_curr = np.angle(X[:, i])
        phi_last = np.angle(Y[:, i - 1])

        #Calculates ideal next phase
        phi_pred = phi_last + omega * delta_t
        
        #Diference between phases and normalization
        phi_error = phi_curr - phi_pred
        phi_error = normalize_phase(phi_curr - phi_pred)

        #Calculate Instantanious frequency
        IF_w = omega + (phi_error/delta_t)
        
        #Phase modification using IF
        phi_mod = np.angle(Y[:, i - 1]) + (IF_w * Hs/fs)

        #Add modify frame to output stft
        Y[:, i] = np.abs(X[:, i]) * np.exp(2*np.pi * 1j * phi_mod) 

    y = istft(Y, hop_length=Hs, win_length=N, n_fft=N, window=win_type)
    
    return y
