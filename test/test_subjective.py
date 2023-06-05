import numpy as np

from utils.wav_utils import read_wav
import plotting
import signals
import pv

#Test PV subjective
N = 2048
Hs = N//4
alpha = 0.6
fs = 22050

test, _ = read_wav("data\\audio_003.wav", fs)
sig_mod = pv.TSM_PV(test, fs, N, alpha, Hs)

plotting.compare_3_results(test, sig_mod, test, fs)