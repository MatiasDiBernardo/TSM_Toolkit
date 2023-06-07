import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import get_window

from utils.wav_utils import read_wav, save_wav
from utils.windows import window_normalization

def normalize_phase(phi_diffrenece):
    """Normalice the value of difference between
    the real and the predicted phase.

    Args:
        phi_diffrenece (np.array): Phase difference.
    Returns:
        (np.array): Normalize phase array.
    """
    normalize_phase = np.zeros(len(phi_diffrenece))
    max_phase = np.pi
    
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
    y[:N] = x[:N]  #First frame
    Ha = int(Hs/alpha)
    cicles = int((len(x) - N)/Ha)  #Amount of frames in the signal.
    last_phase = np.angle(fft(x[:N]))
    omega = np.arange(0,N)*fs/N
    delta_t = Ha/fs

    for m in range(1, cicles):
        #Segment frame and transform it to frequency.
        Xm = x[m * Ha: N + (m * Ha)]
        Xk = fft(Xm*w)
        
        #Calculate Inst Freq
        current_phase = np.angle(Xk)
        phi_error = current_phase - last_phase - (omega * delta_t)
        phi_error = normalize_phase(phi_error)
        IF_w = omega + (phi_error/delta_t)

        #Modify phase and adjust spectrum
        phi_mod = last_phase + IF_w * Hs/fs
        X_mod = np.abs(Xk) * np.exp(1j * 2*np.pi * phi_mod)  #Uses current phi mod for update

        #Rest phase value
        last_phase = phi_mod  #Update the current modify phase

        #Transform to time and relocate in the synthesis frame.
        Xm_mod = ifft(X_mod)
        Xm_mod = np.real(Xm_mod)
        #Xm_mod = np.concatenate([Xm_mod[len(Xm_mod)//2:] , Xm_mod[:len(Xm_mod)//2]])  #Para test

        y[m * Hs: N + (m * Hs)] += (Xm_mod*w)/w_norm #Supuestamente es dividir w_norm pero no queda
        
    return y

def quick_test(path, N, alpha, Hs):
    fs = 22050
    x, _ = read_wav(path, fs)
    rta = TSM_PV(x, fs, N, alpha, Hs)

    save_wav(rta, fs, "data\\quick_test3.wav")

"""
Si uso fs igual 22050 y una ventana de 2048 tengo una longitud de
93ms. 
"""
test_audio = "data\\audio_003.wav" 
N = 2048
Hs = N//2
alpha = 1.5

quick_test(test_audio, N, alpha, Hs)
