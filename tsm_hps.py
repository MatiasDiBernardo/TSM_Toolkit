import numpy as np
import pv
import hps
import ola

def match_sizes_with_padding(x1, x2):
    if len(x1) > len(x2):
        dif = len(x1) - len(x2)
        x2 = np.concatenate([np.zeros(dif), x2])
        return x1, x2
    else:
        dif = len(x2) - len(x1)
        x1 = np.concatenate([np.zeros(dif), x1])
        return x1, x2

def TSM_HPS(x, cfg_ola, cfg_pv, cfg_hps):
    """Mix OLA TSM for precussive signal and PV TSM for
    harmonic signals with HPS separation.

    Args:
        x (np.array): Audio signal.
        fs (int): Sample rate.
        cfg_ola (dict): Config for ola.
        cfg_pv (dict): Config for pv.
        cfg_hps (dict): Config por HPS.
    """

    x_perc, x_harm = hps.hps(x, **cfg_hps)
    x_harm = pv.TSM_PV(x_harm, **cfg_pv)
    x_perc = ola.TSM_OLA(x_perc, **cfg_ola)

    x_perc, x_harm = match_sizes_with_padding(x_perc, x_harm)
    x_output = x_harm + x_perc

    return x_output
