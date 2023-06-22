import numpy as np
from librosa import stft, istft
from scipy.signal import find_peaks

def phase_locking(w, phi_offset, peaks, delta_t):
    """Adjust phase changes relative to spectral peak.

    Args:
        w (np.array): Range of frequencies.
        phi_offset (np.array): Phase diference.
        peaks (np.array): Array of indices with location of peaks.
        delta_t (float): Time diference between frames.

    Returns:
        IF_w: Instantaneous frequency with phase locking.
    """

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
                
        if index_peak == 0:  #Base case
            w_update = w[peaks[index_peak]] + (phi_offset[index_peak]/delta_t)
        
        if index_peak == len(peaks):  #Last case
            w_update = w[peaks[index_peak - 1]] + (phi_offset[index_peak - 1]/delta_t)
            
        IF_w[i] = w_update

    return IF_w

def TSM_PV_FL(x, fs, N, alpha, Hs):
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

        #Find peaks on magnitude spectrum
        frame_mean = np.mean(np.abs(X[:,i]))
        peaks, _ = find_peaks(np.abs(X[:,i]), height=frame_mean, width=2)

        phi_curr = np.angle(X[:, i])
        phi_last = np.angle(Y[:, i - 1])
        phi_pred = phi_last + omega * delta_t
        
        phi_error = phi_curr - phi_pred
        phi_error = np.mod(phi_error + 0.5, 1) - 0.5 

        IF_w = phase_locking(omega, phi_error, peaks, delta_t)
        print(IF_w)

        phi_mod = np.angle(Y[:, i - 1]) + (IF_w * Hs/fs)

        #Phase prop, otra opción que funca bien pero con Hs/6 no funca
        phase_prop += (np.pi * (omega + np.abs(X[:,i]))/fs) * Ha

        Y[:, i] = np.abs(X[:, i]) * np.exp(1j * 2*np.pi * phi_mod) 

    y = istft(Y, hop_length=Hs, win_length=N, n_fft=N, window=win_type)
    
    return y