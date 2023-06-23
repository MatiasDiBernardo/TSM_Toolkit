import numpy as np
from scipy.signal import get_window
from pytsmod import phase_vocoder
from sine_model.sineModel import sineModelAnal, sineModelSynth
from sine_model.sineTransformations import sineTimeScaling

from utils.wav_utils import read_wav, save_wav
from test import plotting
import pv
import pv_pl
import ola
import tsm_hps

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

def apply_algo(x, type, cfg):
    if type == "OLA":
        y = ola.TSM_OLA(x, **cfg)
    if type == "PV":
        y = pv.TSM_PV(x, **cfg)
    if type == "PV_FL":
        y =  pv_pl.TSM_PV_FL(x, **cfg)
    if type == "HPS":
        y = tsm_hps.TSM_HPS(x, **cfg)
    if type == "PV_REF":
        y = phase_vocoder(x, s=cfg["alpha"], win_type="hann", win_size=cfg["N"], syn_hop_size=cfg["Hs"])
    if type == "SIN_M":
        y = Sine_Model(x, **cfg)
    return y

def compare_algorithms(path_audio, algo_comp, plot, save_audios, return_audios):
    """Compare a real audio with different algorithms or diferent parameteres.
    Algo types: "OLA" : Overlapp-add
                "PV" : Phase Vocoder
                "PV_FL" : Phase Vocoder with Phase Locking
                "HPS" : TSM with harmonic-percussive separation
                "PV_REG" : Pytsmod Implementation Phase Vocoder
                "SIN_M" : TSM base on Sine Model.

    Args:
        path_audio (str): Path to audio file.
        algo_comp (list): List with type of algo, config and title. Ex ["PV", cgf1, "PV Cfg1"]
        plot (boolean): Plot or not.
        save_audios (bolean): Save audios or not.
        return_audios (bolean): Return audios or not.
    """

    fs = 22050
    test, _ = read_wav(path_audio, fs, mono=True)

    titles = []
    signals = []

    for data in algo_comp:
        y = apply_algo(test, data[0], data[1])
        signals.append(y)
        titles.append(data[2])
        
    if plot:
        plotting.compare_results(fs, titles, *signals)
    
    if save_audios:
        name = path_audio.split(".")[0]
        for i in range(len(signals)):
            save_wav(signals[i], fs, f"{name}_{titles[i]}.wav" )
    
    if return_audios:
        return signals
 
#Test subjective
path_audio = "audios\sharp_bells.wav"
alpha = 1.5
fs = 22050

cfg_pv = {"N": 2048, "Hs": 2048//4, "alpha": alpha, "fs": fs}
cfg_ola = {"N": 1024, "Hs": 1024//2, "alpha": alpha}
cfg_hps = {"N": 1024, "M":17}

#compare_algorithms(path_audio, plot=True, save_audios=True, cfg_pv=cfg_pv, cfg_ola=cfg_ola, cfg_hps=cfg_hps)

