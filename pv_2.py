import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import get_window
from librosa import stft, istft

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

def instantaneous_frequency(Xk, phi_pred, m, fs, Ha):
    """Calculates the instantaneous frequency of a current frequency frame. 
    And also predicts the phase for the next frame.

    Args:
        Xk (np.array): Frequency spectrum in one frame time.
        phi_pred (float): Predicted phase of the current frame.
        m (int): Frame index.
        Fs (int): Sample rate.
        Ha (int): Analisis hop size.
    Returns:
        (np.array): Instanteneous frequency.
        (np.array): Predicted phase for next frame.
    
    """
    #Calculate necessary parameters
    N = len(Xk)
    t1 = (m*Ha)/fs
    t2 = ((m + 1)*Ha)/fs
    delta_t = t2 - t1  #Same as Ha/fs
    k = np.arange(0,N)
    w = k * fs/N
    #w = 2 * np.pi * k/N

    #Calculate current and next phase
    phi_real = np.angle(Xk)
    phi_pred_next_frame = phi_real + w*delta_t

    #Adjust frequencies on the current frame
    phi_offset = normalize_phase(phi_real - phi_pred)
    IF_w = w + (phi_offset/delta_t)

    return IF_w, phi_pred_next_frame

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
    Ha = int(Hs/alpha)
    cicles = int((len(x) - N)/Ha) + 1  #Amount of frames in the signal.
    pred_phase = 0
    mod_phase = 0
    Xf = stft(x, n_fft=N, hop_length=Ha, win_length=N)
    X_mod = []
    for i in range(Xf.shape[0]):
        frame = Xf[i, :]
        IF_w, next_phase_pred = instantaneous_frequency(frame, pred_phase, i, fs, Ha)
        frame_mod = np.abs(frame) * np.exp(1j * 2*np.pi * mod_phase)
        X_mod.append(frame_mod)

        #Resets values for next iteration
        pred_phase = next_phase_pred
        mod_phase = mod_phase + IF_w * Ha/fs
    
    X_mod = np.array(X_mod)
    y = istft(X_mod, hop_length=Hs, win_length=N, n_fft=N)

    return y

def quick_test(path, N, alpha, Hs):
    fs = 22050
    x, _ = read_wav(path, fs)
    rta = TSM_PV(x, fs, N, alpha, Hs)

    save_wav(rta, fs, "data\\quick_test8.wav")

"""
Si uso fs igual 22050 y una ventana de 2048 tengo una longitud de
93ms. 
"""
test_audio = "data\\sunny-original.flac" 
N = 2048
Hs = N//4
alpha = 1.5

quick_test(test_audio, N, alpha, Hs)
