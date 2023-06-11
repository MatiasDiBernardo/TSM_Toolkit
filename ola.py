#Implementacion OLA 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from utils.wav_utils import read_wav, save_wav
from utils.windows import window_normalization

def TSM_OLA(x, N, alpha, Hs): 
    """Applies TSM procedure based on OLA: Overlapp and Add.

    Args:
        x (np.array): Audio signal.
        N (int): Window lenght.
        alpha (float): Stretching factor.
        Hs (int): Synthesis hopsize length.

    Returns:
        np.array: Time modify audio signal.
    """

    #defining output vector size according to scale factor
    if alpha == 1:
        return x 
    if alpha < 1: 
        y = np.zeros(int(len(x) * alpha) + N)  #N accounts for last frame.
    if alpha > 1:
        y = np.zeros(int(len(x) * alpha)) 
    
    #Window election
    win_type = "hann"
    w = get_window(win_type, N)
    w_norm = window_normalization(w, Hs) 

    #Inicialization
    Ha = int(Hs/alpha)
    cicles = int((len(x) - N)/Ha) + 1  #Amount of frames in the signal.

    for m in range(cicles):

        #segment input into analysis frames 
        x_m = x[m * Ha: N + (m * Ha)]

        #compute output signal by windowing x_m and normalizing
        #and overlapping according to Hs
        y[m * Hs: N + (m * Hs)] += (x_m*w)/(w_norm) 
                
    return y 


#funcion de quick test OLA
def quick_test_OLA(path, N,fs, alpha, Hs,savename):
    '''
    Computes the TSM with the OLA procedure to a given signal audio file.

    Parameters:

    path: directory of the audio sample
    N (int): Number of samples for the window
    fs (int): Sample rate
    alpha (float): Stretching factor.
    Hs (int): Synthesis hopsize length.
    savename (boolean): Optional. Necessary for saving the file.
    '''
    
    x, _ = read_wav(path, fs)
    x_out = TSM_OLA(x,N,alpha,Hs)

    if savename:
        nameout = "audios_mod/prueba_ola.wav"
        save_wav(x_out, fs, nameout)

    return x, x_out 

'''
Observacion:
tamaño de ventana en segundos: w_size = N/fs
Para fs = 44100

algunos casos:
Si N = 1024 -> w_size = 23 ms
Si N = 2048 -> w_size = 46 ms
Si N = 4096 -> w_size = 92 ms
Si N = 8192 -> w_size = 184 ms
'''

#simple initial tests

'''
PRUEBA1 - signal: synth - fs=44100 - N=4096 - alpha=1.5 - Hs=N/2
'''
#quick_test_OLA('audios/synth.wav',4096,44100,1.5,4096//2,'synth_prueba1')

'''
PRUEBA2 - signal: sharp_bells - fs=44100 - N=4096 - alpha=1.5 - Hs=N/2
'''
#quick_test_OLA('audios/sharp_bells.wav',4096,44100,1.5,4096//2,'bells_prueba1')


