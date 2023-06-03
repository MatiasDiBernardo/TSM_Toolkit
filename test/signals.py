import numpy as np

"""
Aca hay que ir definienido las funciones que son importantes para comparar según el método.
Este script es el que tiene mas sentido de armarlo como una sola clase pero por ahora 
vamos a dejarlo así y después vemos. 
"""

def simple_sine(f0, fs, alpha):
    """Generates a base cosine signal and a modifed signal
    according to the stretching factor.

    Args:
        f0 (float): Main frequency.
        fs (int): Sample rate
        alpha (float): Stretching factor.
    Return:
        (np.array): Base case cosine function.
        (np.array): Modify cosine function
    """
    t_base = np.linspace(0, 1, fs)
    t_ideal = np.linspace(0, alpha, int(fs*alpha))
    signal_base = np.cos(2*np.pi*f0*t_base)
    signal_ideal = np.cos(2*np.pi*f0*t_ideal)

    return signal_base, signal_ideal

def fast_changes(fluctiation_time, alpha):
    """Generates signal with fast frequency changes.

    Args:
        fluctiation_time (float): Time to exchanges the frequencies present
        in the signal.
        alpha (float): Time for the modified ideal version.
    """
    return None