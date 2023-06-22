import numpy as np
from scipy.signal import get_window
from pytsmod import phase_vocoder
from sine_model.sineModel import sineModelAnal, sineModelSynth
from sine_model.sineTransformations import sineTimeScaling

from utils.wav_utils import read_wav, save_wav
import plotting
import pv
import pv_2
import pv_pl
import ola

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

def compare_algorithms(path_audio, plot, save_audios, cfg_pv, cfg_ola):
    fs = 22050
    test, _ = read_wav(path_audio, fs)

    #Calculate diferent algotithms on the same audio
    pv_mod = pv_2.TSM_PV_copy(test, **cfg_pv)
    #pv_mod = pv_2.TSM_PV_copy(test, **cfg_pv)
    pv_fl_mod = pv_pl.TSM_PV_FL(test, **cfg_pv)
    pv_ref = phase_vocoder(test, s=cfg_pv["alpha"], win_type="hann", win_size=cfg_pv["N"], syn_hop_size=cfg_pv["Hs"])
    sin_model = Sine_Model(test, **cfg_pv)
    ola_mod =  ola.TSM_OLA(test, **cfg_ola)
    
    if plot:
        #La idea de este va a ser comparar los resultados de nuestra implementación, contra la de referencia, contra la de
        #el sine model que es otro método que lo saqué de otro lado.
        titles = ["Original", "TSM Nuestro", "TSM Librería", "Sine Model"]
        plotting.compare_results(fs, titles, test, pv_mod, pv_ref, sin_model)
    
    if save_audios:
        name = path_audio.split(".")[0]
        save_wav(pv_mod, fs, name + "_PV.wav" )
        save_wav(pv_fl_mod, fs, name + "_PV_FL.wav" )
        save_wav(pv_ref, fs, name + "_PV_REF.wav")
        save_wav(sin_model, fs, name + "_SIN.wav")
        save_wav(ola_mod, fs, name + "_OLA.wav")

#Test subjective
path_audio = "audios\sharp_bells.wav"
alpha = 1.5
fs = 22050

cfg_pv = {"N": 2048, "Hs": 2048//4, "alpha": alpha, "fs": fs}
cfg_ola = {"N": 1024, "Hs": 1024//2, "alpha": alpha}

#compare_algorithms(path_audio, plot=True, save_audios=True, cfg_pv=cfg_pv, cfg_ola=cfg_ola)

