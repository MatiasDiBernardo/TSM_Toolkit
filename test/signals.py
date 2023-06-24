import numpy as np
from scipy import signal

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

def simple_impulse(fs, alpha, time = 1):
    """Generates a base impulse signal and a modifed signal
    according to the stretching factor. duty cycle =.5

    Args:
        fs (int): Sample rate
        alpha (float): Stretching factor.
        time (float): Time of the signal in seconds. Default=1.
    Return:
        (np.array): Base case cosine function.
        (np.array): Modify cosine function
    """
    t_base = np.linspace(0, time, int(fs*time))
    t_ideal = np.linspace(0, time * alpha, int(fs*alpha*time))

    signal_base = np.abs(signal.square(2*np.pi*5000*t_base, duty=0.25))
    signal_ideal = np.abs(signal.square(2*np.pi*5000*t_ideal, duty=0.25))

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

def impulses(N, fs, A, t, time = 1):

    """Generates a series impulses.
    Args:
        N (int): number of impulses.
        fs (int): Sample rate.
        A(float): Amplituf
        t(int): lenght of the impulse in miliseconds.
        time (float): Time of the signal in seconds. Default=1.
    Return:
        (np.array): Vector of impulses.
        fs(int): sample rate.
    """
    t = t/1000
    T = 1/fs
    nt = int(t/T)
    t_base = np.linspace(0, time, int(fs*time))
    x = np.zeros(len(t_base))
    t_s =  time/(N+1)  
    if t_s<=t:
        return print('The duration of the impuse exceeds the time interval between impulses')
    t_i = np.arange(t_s,1,t_s)
    n = t_i/T
    n_i = np.arange(0,nt,1)
    imp_exp = A**(n_i/(len(n_i)-1))
    

    for i in range(len(n)):
        for l in range(nt):
            x[ int(n[i])+ l] = imp_exp[l]
    return x

def harmonic(N, f0, fs, A, time = 1):

    """Generates an armonic signal with frecuency f, and N armonics.
    Args:
        N (int): number of impulses.
        f0 (float): Frecuency of the fundamental.
        fs (int): Sample rate.
        M (int): lenght of the impulse in miliseconds.
        time (float): Time of the signal in seconds. Default=1.
    Return:
        (np.array): output signal.
        fs(int): sample rate.
    """

    t_base = np.linspace(0, time, int(fs * time))
    x = np.zeros(len(t_base))
    A = abs(A)
    if A>1: 
        for i in range(N):
            x = x + (A**(1-(i/20))) *np.sin(2*np.pi *(f0)* t_base* (i+1))
 
    else:
        for i in range(N):
            x = x + (A**(1+(i/20))) *np.sin(2*np.pi *(f0)* t_base* (i+1))
    return x
