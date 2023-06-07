import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import get_window
from scipy.signal import find_peaks

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

def phase_locking(w, phi_offset, peaks, delta_t):

    current_peak = peaks[0]  #Tracks current peak value
    index_peak = 0  #Tracks peak index
    IF_w = np.zeros(len(w))
    
    for i in range(len(w)):
        if i > current_peak:  #Updates if index is grater than current peak
            index_peak += 1
            if index_peak < len(peaks):  #Check for last peak
                current_peak = peaks[index_peak]
            
        if index_peak >= 1 and index_peak < len(peaks):  #Avoids checking first and last peak
            #Compares distance between index and closest peaks
            dif_before = i - peaks[index_peak - 1]
            dif_after = peaks[index_peak] - i
            
            if dif_before <= dif_after:  #Belongs to previous peak
                w_update = w[peaks[index_peak - 1]] + (phi_offset[index_peak - 1]/delta_t)
            else:  #Uses current peak
                w_update = w[peaks[index_peak]] + (phi_offset[index_peak]/delta_t)
                
        if index_peak == 0:
            w_update = w[peaks[index_peak]] + (phi_offset[index_peak]/delta_t)
        
        if index_peak == len(peaks):
            w_update = w[peaks[index_peak - 1]] + (phi_offset[index_peak - 1]/delta_t)
            
        IF_w[i] = w_update

    return IF_w

def instantaneous_frequency(Xk, phi_pred, m, fs, Ha, peaks):
    """Calculates the instantaneous frequency of a current frequency frame. 
    And also predicts the phase for the next frame.

    Args:
        Xk (np.array): Frequency spectrum in one frame time.
        phi_pred (float): Predicted phase of the current frame.
        m (int): Frame index.
        Fs (int): Sample rate.
        Ha (int): Analisis hop size.
        peaks (np.array): Array with indices corresponding to
        peaks location in the magnitude spectrum.
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

    #Calculate current and next phase
    phi_real = np.angle(Xk)
    phi_pred_next_frame = phi_real + w*delta_t

    #Adjust frequencies on the current frame
    phi_offset = normalize_phase(phi_real - phi_pred)

    #Phase locking
    if len(peaks) > 0:
        IF_w = phase_locking(w, phi_offset, peaks, delta_t)
    else:
        IF_w = w + (phi_offset/delta_t)
        
    #Posible re factor para que si peaks = 0, llamar de nuevo a peaks detection pero con otros
    #parámetros. Implementación como clase se vuelve mas relevante. 
    
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

    for m in range(cicles):
        #Segment frame and transform it to frequency.
        Xm = x[m * Ha: N + (m * Ha)]
        Xk = fft(Xm*w)
        mag = np.abs(Xk)
        mean = np.mean(mag)

        #Tengo que definir que parámetros van mejor
        peaks, _ = find_peaks(np.abs(Xk), height=mean, width=2)
        #plt.plot(peaks, mag[peaks], "x")
        #plt.plot(np.ones(len(mag)) * mean, "--", color="gray")
        #plt.plot(mag)
        #plt.show()
        
        #Modify frequency frame by shifting the phase.
        IF_w, next_phase_pred = instantaneous_frequency(Xk, pred_phase, m, fs, Ha, peaks)

        #Resets values for next iteration
        pred_phase = next_phase_pred
        mod_phase = mod_phase + IF_w * Hs/fs
        X_mod = np.abs(Xk) * np.exp(1j * 2*np.pi * mod_phase)  #Aca es así o np.angle(mod_phase)

        #Transform to time and relocate in the synthesis frame.
        Xm_mod = ifft(X_mod)
        Xm_mod = np.real(Xm_mod)
        #Xm_mod = np.concatenate([Xm_mod[len(Xm_mod)//2:] , Xm_mod[:len(Xm_mod)//2]])  #Para test

        y[m * Hs: N + (m * Hs)] += (Xm_mod*w)*w_norm #Supuestamente es dividir w_norm pero no queda
        
    return y

def quick_test(path, N, alpha, Hs):
    fs = 22050
    x, _ = read_wav(path, fs)
    rta = TSM_PV(x, fs, N, alpha, Hs)

    save_wav(rta, fs, "data\\quick_test4.wav")

"""
Si uso fs igual 22050 y una ventana de 2048 tengo una longitud de
93ms. 
"""
test_audio = "data\\audio_003.wav" 
N = 2048
Hs = N//4
alpha = 1.5

quick_test(test_audio, N, alpha, Hs)
