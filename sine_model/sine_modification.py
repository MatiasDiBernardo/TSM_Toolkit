import numpy as np
from scipy.signal import get_window
from sine_model.sineModel import sineModelAnal, sineModelSynth
from sine_model.sineTransformations import sineTimeScaling

def Sine_Model(x, N=2048, Hs= 2048//4, alpha= 1.2, fs=22050):
    w = get_window("hann", N)
    H = int(Hs/alpha)
    H = 512
    t = -100  #Threshold tolerance
    time_scaling = np.array([0, 0, 1, alpha])  #Apply constant scaling factor

    x_freq, x_mag, x_phase = sineModelAnal(x, fs, w, N, H, t) 
    y_freq, y_mag = sineTimeScaling(x_freq, x_mag, time_scaling)
    y = sineModelSynth(y_freq, y_mag, np.array([]), N, H, fs)  #Empty fase is handled with phase propagation

    return y