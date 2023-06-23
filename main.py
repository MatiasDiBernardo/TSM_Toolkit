import numpy as np

from utils.wav_utils import read_wav, save_wav, is_stereo
from pytsmod import phase_vocoder
from sine_model.sine_modification import Sine_Model
import ola
import pv
import pv_pl
import tsm_hps
import hps

FS = 22050  #No se si conviene dejarlo así o no

def apply_algo(x, type, cfg):
    if type == "OLA":
        y = ola.TSM_OLA(x, **cfg)
    if type == "PV":
        y = pv.TSM_PV(x, **cfg)
    if type == "PV_FL":
        y =  pv_pl.TSM_PV_FL(x, **cfg)
    if type == "HPS_SOLO":
        y = hps.hps(x, **cfg)
    if type == "HPS":
        y = tsm_hps.TSM_HPS(x, *cfg)
    if type == "PV_REF":
        y = phase_vocoder(x, s=cfg["alpha"], win_type="hann", win_size=cfg["N"], syn_hop_size=cfg["Hs"])
    if type == "SIN_M":
        y = Sine_Model(x, **cfg)
    return y

def define_cfg(type, cfg_ola, cfg_pv, cfg_hps_solo, cfg_hps):
    if type == "OLA":
        cfg = cfg_ola
    if type == "PV":
        cfg = cfg_pv
    if type == "PV_FL":
        cfg = cfg_pv
    if type == "HPS_SOLO":
        cfg = cfg_hps_solo
    if type == "HPS":
        cfg = cfg_hps
    if type == "PV_REF":
        cfg = cfg_pv
    if type == "SIN_M":
        cfg = cfg_pv
    return cfg

def main(file_path, alpha, type):
    """Main function that contains all the functionality in this repository to
    perform time-scale-modifications. With the predefine setting that work well
    for each algorithm.
    Algo types: "OLA" : Overlapp-add
                "PV" : Phase Vocoder
                "PV_FL" : Phase Vocoder with Phase Locking
                "HPS_SOLO" : Harmonic-percussive separation
                "HPS" : TSM with harmonic-percussive separation
                "PV_REG" : Pytsmod Implementation Phase Vocoder
                "SIN_M" : TSM base on Sine Model.

    Args:
        file_path (string): Path to the target audio file to modify.
        alpha (float): Stretching factor.
        type (string): Type of algorithm to implement.
    """
    #Agregar chequeo de la dimensiones de alpha y ver hasta donde se la banca

    #Completar con las mejores configs

    cfg_ola = {"N": 1024, "Hs": 1024//2, "alpha": alpha}
    cfg_pv = {"N": 4096, "Hs": 4096//4, "alpha": alpha, "fs": 22050}
    cfg_hps_solo = {"N": 1024, "M" : 17}
    cfg_hps = [cfg_ola, cfg_pv, cfg_hps_solo]

    cfg = define_cfg(type, cfg_ola, cfg_pv, cfg_hps_solo, cfg_hps)
    
    if is_stereo(file_path):
        xl, xr = read_wav(file_path, FS)
        yl = apply_algo(xl, type, cfg)
        yr = apply_algo(xr, type, cfg)

        #Checkear que formato es el que va
        y = np.array([yl, yr])  #Channels, then audio
        y = np.reshape(y, (y.shape[1], 2))  #Audio, then channels
    else:
        x, _ = read_wav(file_path, FS)
        y = apply_algo(x, type, cfg_ola, cfg_pv, cfg_hps)
    
    return y
