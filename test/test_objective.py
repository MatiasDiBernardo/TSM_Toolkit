import numpy as np

from utils.wav_utils import read_wav
import plotting
import signals
import pv


#Test PV

#Parametres
N = 2048
Hs = N//4
alpha = 1.4
fs = 22050

x_base, x_ideal = signals.simple_sine(50, fs, alpha)
x_tsmpv = pv.TSM_PV(x_base, fs, N, alpha, Hs)

plotting.compare_3_results(x_base, x_ideal, x_tsmpv, fs)
