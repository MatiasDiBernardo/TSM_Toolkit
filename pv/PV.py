import numpy as np
import matplotlib.pyplot as plt
from utils.wav_utils import read_wav, save_wav
from scipy.fftpack import fft, ifft
from scipy.signal import get_window

def normalize_phase(phi_diffrenece):
    """Normalice the value of difference between
    the real and the predicted phase.

    Args:
        phi_diffrenece (float): Phase difference

    """
    
    normalize_phase = np.zeros(len(phi_diffrenece))
    #for i, phi in enumerate(phi_diffrenece):
    #    if phi > 0.5:
    #        normalize_phase[i] = 0.5
    #        continue
    #    if phi < -0.5:
    #        normalize_phase[i] = -0.5
    #        continue
    #    normalize_phase[i] = phi
    
    for i, phi in enumerate(phi_diffrenece):
        if phi > np.pi:
            normalize_phase[i] = phi % np.pi
            continue
        if phi < -np.pi:
            normalize_phase[i] = -1 * ((-1*phi)%np.pi)
            continue
        normalize_phase[i] = phi
    return normalize_phase
    

def instantanius_frequency(Xk, phi_pred, m, fs, Ha):
    """Calculates the instantanius frequency of a current frequency frame. 
    And also predicts the phase for the next frame.

    Args:
        Xk (np.array): Frequency spectrum in one frame time.
        phi_pred (float): Predicted phase of the current frame.
        m (int): Frame index.
        Fs (int): Sample rate.
        Ha (int): Analisis hop size.
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
    IF_w = w + (phi_offset/delta_t)
    print("Comparo w y su clon: ", w, IF_w)

    return IF_w, phi_pred_next_frame

def phase_vocoder(x, fs, N, alpha, Hs):

    if alpha == 1:
        return x
    if alpha < 1:
        y = np.zeros(int(len(x) * alpha) + N)  #N accounts for last frame.
    if alpha > 1:
        y = np.zeros(int(len(x) * alpha)) 
    
    Ha = int(Hs/alpha)
    win_type = "blackman"
    
    cicles = int((len(x) - N)/Ha) + 1  #Amount of frames in the signal.
    pred_phase = 0
    mod_phase = 0

    for m in range(cicles):
        #Segment frame and transform it to frequency.
        Xm = x[m * Ha: N + m * Ha]
        w = get_window(win_type, N)
        #plt.plot(Xm*w)
        Xk = fft(Xm * w)

        #Modify frequency frame by shifting the phase.
        IF_w, next_phase_pred = instantanius_frequency(Xk, pred_phase, m, fs, Ha)
        X_mod = np.abs(Xk) * np.exp(1j * 2*np.pi * mod_phase)
        print("Mag iguales: ", np.allclose(np.abs(Xk), np.abs(X_mod)))


        #Resets values for next iteration
        pred_phase = next_phase_pred
        mod_phase = mod_phase + IF_w * Ha/fs

        #Transform to time and relocate in the synthesis frame.
        Xm_mod = ifft(X_mod)
        Xm_mod = np.real(Xm_mod)
        Xm_mod = np.concatenate([Xm_mod[len(Xm_mod)//2:] , Xm_mod[:len(Xm_mod)//2]])  #Porque?
        
        #plt.plot(Xm_mod)
        #plt.show()

        y[m * Hs: N + m * Hs] += Xm_mod * w
        #plt.plot(y)
        #plt.show()
        
    return y

def quick_test(path, N, alpha, Hs):
    
    fs = 22050
    x, _ = read_wav(path, fs)

    rta = phase_vocoder(x, fs, N, alpha, Hs)

    save_wav(rta, fs, "data\\quick_test4.wav")

"""
Si uso fs igual 22050 y una ventana de 2048 tengo una longitud de
93ms. 
"""

N = 2048
Hs = N//4
alpha = 0.8

quick_test("data\\audio_003.wav", N, alpha, Hs)

