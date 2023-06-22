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

def TSM_PV_copy(x, fs, N ,alpha, Hs):

    #Window election
    win_type = "hann"
    w = get_window(win_type, N)

    #Inicialization
    Ha = int(Hs/alpha)
    X = stft(x, n_fft=N, hop_length=Ha, win_length=N)
    Y = np.zeros(X.shape, dtype=complex)
    Y[:, 0] = X[:, 0]  # phase initialization

    k = np.arange(N / 2 + 1)  #Stft uses only postive side
    omega = 2 * np.pi * k / N  #From 0 to pi

    for i in range(1, X.shape[1]):
        dphi = omega * alpha

        #plt.plot(np.real(ifft(X[:,i])))
        ph_curr = np.angle(X[:, i])
        ph_last = np.angle(X[:, i - 1])

        hpi = (ph_curr - ph_last) - dphi  #Dif phi vs phi pred
        hpi = hpi - 2 * np.pi * np.round(hpi / (2 * np.pi))  #Unwrap phase entre -pi y pi

        ipa_sample = (omega + hpi / alpha)  #IF

        ipa_hop = ipa_sample * Hs

        ph_syn = np.angle(Y[:, i - 1])
        theta = ph_syn + ipa_hop - ph_curr
        phasor = np.exp(1j * theta)

        Y[:, i] = phasor * X[:, i]

        #plt.plot(np.real(ifft(Y[:,i])))
        #plt.show()
    
    y = istft(Y, hop_length=Hs, win_length=N, n_fft=N)
    
    return y

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

    #Inicialization
    Ha = int(Hs/alpha)
    delta_t = Ha/fs
    win_type = "hann"
    X = stft(x, n_fft=N, hop_length=Ha, win_length=N, window=win_type)
    Y = np.zeros(X.shape, dtype=complex)
    Y[:, 0] = X[:,0]  # phase initialization

    k = np.arange(N / 2 + 1)  #Stft uses only postive side
    omega = k * fs/N  #From 0 fs/2

    phase_prop = 2 * np.pi * np.random.rand(X[:, 0].size)  # initialize synthesis phases
    for i in range(1, X.shape[1]):
        phi_curr = np.angle(X[:, i])
        phi_last = np.angle(Y[:, i - 1])
        phi_pred = phi_last + omega * delta_t
        
        phi_error = phi_curr - phi_pred
        phi_error = normalize_phase(phi_curr - phi_pred)

        IF_w = omega + (phi_error/delta_t)
        IF_w += np.mean(omega - IF_w)

        phi_mod = np.angle(Y[:, i - 1]) + (IF_w * Hs/fs) - phi_curr

        #Compare two phases
        #plt.plot(np.mod(phi_mod, 2*np.pi)-np.pi)
        #plt.plot(phi_curr)
        #plt.show()

        #plt.plot(omega)
        #plt.plot(IF_w)
        #plt.show()
        
        #Phase prop, otra opción pero funca bien pero con Hs/6 no funca
        phase_prop += (np.pi * (omega + np.abs(X[:,i]))/fs) * Ha
        #ytphase += (np.pi * (lastytfreq + tfreq[l, :]) / fs) * H  # propagate phases

        Y[:, i] = np.abs(X[:, i]) * np.exp(1j * 2*np.pi * phi_mod) 

    y = istft(Y, hop_length=Hs, win_length=N, n_fft=N, window=win_type)
    
    return y
    
def quick_test(path, N, alpha, Hs):
    fs = 22050
    x, _ = read_wav(path, fs)
    rta = TSM_PV(x, fs, N, alpha, Hs)

    save_wav(rta, fs, "data\\quick_test2.wav")

"""
Si uso fs igual 22050 y una ventana de 2048 tengo una longitud de
93ms. 
"""
test_audio = "data\SingingVoice_ORIG.wav" 
#test_audio = "data\\audio_003.wav" 
N = 2048
Hs = N//4
alpha = 0.5

#quick_test(test_audio, N, alpha, Hs)