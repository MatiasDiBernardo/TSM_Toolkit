import numpy as np

from utils.wav_utils import read_wav, save_wav
import plotting
import pv
import pv_phase_locking
import ola

def compare_algotithms(path_audio, plot, save_audios, cfg_pv, cfg_ola):
    fs = 22050
    test, _ = read_wav(path_audio, fs)
    pv_mod = pv.TSM_PV(test, **cfg_pv) 
    pv_fl_mod = pv_phase_locking.TSM_PV_phase_locking(test, **cfg_pv)
    ola_mod =  ola.TSM_OLA(test, **cfg_ola)
    

    if plot:
        plotting.compare_3_results(pv_fl_mod, pv_mod, ola_mod, fs)
    
    if save_audios:
        name = path_audio.split(".")[0]
        save_wav(pv_mod, fs, name + "_PV.wav" )
        save_wav(pv_fl_mod, fs, name + "_PV_FL.wav" )
        save_wav(ola_mod, fs, name + "_OLA.wav")
        

#Test subjetive
path_audio = "audios\sharp_bells.wav"
alpha = 1.4
fs = 22050

cfg_pv = {"N": 2048, "Hs": 2048//4, "alpha": alpha, "fs": fs}
cfg_ola = {"N": 1024, "Hs": 1024//2, "alpha": alpha}

compare_algotithms(path_audio, plot=True, save_audios=True, cfg_pv=cfg_pv, cfg_ola=cfg_ola)

