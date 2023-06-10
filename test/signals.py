import numpy as np

"""
Aca hay que ir definienido las funciones que son importantes para comparar según el método.
Este script es el que tiene mas sentido de armarlo como una sola clase pero por ahora 
vamos a dejarlo así y después vemos. 
"""

def simple_sine(f0, fs, alpha, time = 1):
    """Generates a base cosine signal and a modifed signal
    according to the stretching factor.

    Args:
        f0 (float): Main frequency.
        fs (int): Sample rate
        alpha (float): Stretching factor.
        time (float): Time of the signal in seconds. Default=1.
    Return:
        (np.array): Base case cosine function.
        (np.array): Modify cosine function
    """
    t_base = np.linspace(0, time, int(fs*time))
    t_ideal = np.linspace(0, time * alpha, int(fs*alpha*time))
    signal_base = np.cos(2*np.pi*f0*t_base)
    signal_ideal = np.cos(2*np.pi*f0*t_ideal)

    return signal_base, signal_ideal

def fast_changes(f1, f2, fs, fluctuation_time, alpha, time=1):
    """Generates signal with fast frequency changes in the time domain.

    Args:
        f1 (float): Frequency of first sinusoid.
        f2 (float): Frequency of second sinusoid.
        fs (int): Sample rate.
        fluctuation_time (float): Time in seconds between alternationg sinusoids.
        alpha (float): Stretching factor.
        time (float, optional): Time in seconds of the signals. Defaults to 1.

    Return:
        (np.array): Base case cosine function.
        (np.array): Modify cosine function

    """
    t_base = np.linspace(0, time, int(fs*time))
    t_ideal = np.linspace(0, time * alpha, int(fs*alpha*time))

    x_base = np.zeros(len(t_base))
    for i in range(len(x_base)):
        if (i/fs) % (2*fluctuation_time) < fluctuation_time:
            f = f1
        else:
            f = f2
        x_base[i] = np.cos(2*np.pi *f * t_base[i])

    x_ideal = np.zeros(len(t_ideal))
    for i in range(len(x_ideal)):
        if (i/fs) % (2*fluctuation_time) < fluctuation_time:
            f = f1
        else:
            f = f2
        x_ideal[i] = np.cos(2*np.pi *f * t_ideal[i])

    return x_base, x_ideal