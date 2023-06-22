from utils.wav_utils import read_wav, save_wav, is_stereo
import ola
import pv
import pv_pl
import tsm_hps

FS = 22050  #No se si conviene dejarlo así o no

def main(file_path, type):
    """Main function that contains all the functionality in this repository to
    perform time-scale-modifications. With the predefine setting that work well
    each algorithm.

    Args:
        file_path (string): Path to the target audio file to modify.
        type (string): Type of algorithm to implement.
    """

    #Completar con las mejores configs
    cfg_ola = {}
    cfg_pv = {}
    cfg_hps = {}

    if is_stereo(file_path):
        signal = read_wav(file_path, FS)
        #Hacer lo mismo pero para estereo
    else:
        x, _ = read_wav(file_path, FS)
        if type == "OLA":
            y = ola.TSM_OLA(x, **cfg_ola)
        if type == "PV":
            y = pv.TSM_PV(x, **cfg_pv)
        if type == "PV_FL":
            y =  pv_pl.TSM_PV_FL(x, **cfg_pv)
        if type == "HPS":
            y = tsm_hps.TSM_HPS(x, **cfg_hps)
    
    return y
            