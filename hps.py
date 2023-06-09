#Aca va la implementación del HPS


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import stft 
from scipy.signal import get_window
from scipy.signal import medfilt2d as median
from librosa import stft, istft

from utils.wav_utils import read_wav, save_wav
from utils.windows import window_normalization

#parte de prueba temporal para definir señales
L = 10000
fs = 44100
n = np.arange(0, L-1, 1) 
f1 = 100
f2 = 200
#Componentes armonicas
x = np.sin(n * (f1/fs) * 2 * np.pi ) + np.sin(n * (f2/fs) * 2 * np.pi )
#Impulsos
for i in range(20):# Generamos inpulsos de 20 muestras 
    x[i+300] = 10
    x[i+5000] = 10       

plt.plot(n ,x )
plt.grid()
plt.show()

def HPS(x, fs, N, M):
    """ Divides the percusive and harmonics components from the input signal
    returning two diferent signals.

    Args:
        x (np.array): Audio signal.
        fs (int): Sample rate.
        N (int): Window lenght for FFT.
        M (int): Median filter window leight, this value must be 
        odd. In the contrary it will be forced to the upper inmedate near.

    Returns:
        xp (np.array): Percusive signal.
        xh (np.array): Harmonic signal.
    """
    #Obtain the STFT and the spectogram Y 
    X = stft(x, n_fft = N )
    x1 = istft(X,n_fft = N )
  
    Y = abs(X)

    Mh = median(Y,(1,M)) #2D median filter with window
    Mp = median(Y,(M,1))   
    #for in range(len())
    #if Mh > Mp:
        

    #Auxiliar testing operation Dspues la borro  
    Cp = istft(Mp, n_fft = N)
    Ch = istft(Mh, n_fft = N)

    # Evaluamos para cada frecuencia y segmento 
    a , b = np.shape(Mh)
    for i in range(a):
        for l in range(b):
            if Mh[i,l] > Mp[i,l]:
                Mh[i,l] = 1
            else:
                Mp[i,l] = 1 

    Xp = X * Mp
    Xh = X * Mh

    Xp = istft(Xp, n_fft = N)
    Xh = istft(Xh, n_fft = N)

    return X, Xp, Xh, x1




N = 4096
X , xp, xh, x1 = HPS(x ,fs ,N , 17 )
y = abs(X)

plt.figure() #Señal original
plt.specgram(x, NFFT =N, Fs = fs, noverlap = N//4)
#plt.yscale("log")
plt.ylim([0, 2000])
plt.colorbar()
plt.grid()
plt.show()


plt.figure() #Espectro mascara percusiva
plt.specgram(xp, NFFT = N,Fs = fs)
#plt.yscale("log")
plt.ylim([0, 2000])
plt.grid()
plt.colorbar()
plt.show()

plt.figure() #Espectro mascara armonica
plt.specgram(xh, NFFT = N,Fs = fs)
#plt.yscale("log")
plt.ylim([0, 2000])
plt.grid()
plt.colorbar()
plt.show()



#Waveform
n =  np.arange(0,len(xh))
plt.plot(n, abs(xh))
plt.plot(n, abs(xp))

plt.show()

 # %%


