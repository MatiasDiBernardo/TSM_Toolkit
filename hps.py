
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft 
from scipy.signal import medfilt as mediann
from librosa import stft, istft

def HPS(x, N, M):
    """ Divides the percusive and harmonics components from the input signal
    returning two diferent signals.

    Args:
        x (np.array): Audio signal.
        fs (int): Sample rate.
        N (int): Window lenght for STFT.
        M (int): Median filter window leight, this value must be 
        odd. In the contrary it will be forced to the upper inmedate.

    Returns:
        xp (np.array): Percusive signal.
        xh (np.array): Harmonic signal.
    """
    #Obtain the STFT and the spectogram Y 
    X = stft(x, n_fft = N )
    Y = abs(X)

    Yh = mediann(Y,(1,M)) #2D median filter with window
    Yp = mediann(Y,(M,1))   

    Mh = np.zeros(np.shape(Yh))
    Mp = np.zeros(np.shape(Yh))

    a , b = np.shape(Yh)
    # Evaluamos para cada frecuencia y segmento 
    for i in range(a):
        for l in range(b):
            if Yh[i,l] > Yp[i,l]:
                Mh[i,l] = 1
            else:
                Mp[i,l] = 1 

    Xp = X * Mp
    Xh = X * Mh

    #Inverse transform for each isolated spectrum 
    xp = istft(Xp, n_fft = N)
    xh = istft(Xh, n_fft = N)

    #Compensate the gap created by the median filter
    l = len(x)-len(xp)
    xp = np.concatenate((xp, np.zeros(l)))
    xh = np.concatenate((xh, np.zeros(l)))
  
    return xp, xh

# %% SEÑAL X 

#parte de prueba temporal para definir señales
L = 5000
fs = 88200
n = np.arange(0, L-1, 1) 
f1 = 100
f2 = 400
#Componentes armonicas
x = np.sin(n * (f1/fs) * 2 * np.pi ) + np.sin(n * (f2/fs) * 2 * np.pi )
#Impulsos
for i in range(20):# Generamos inpulsos de 20 muestras 
    x[i+300] = 10
    x[i+3000] = 10       

plt.plot(n ,x )
plt.grid()
plt.show()
# %% USANDO LA FUNCION (BORRAR)

M = 13
N = 2048
xp, xh = HPS(x , N , M)

#Waveform armonica  
n =  np.arange(0,len(xh))

#plt.plot(n, xh)

plt.plot(n, x, 'blue')
plt.plot(n, xh,'orange')
plt.axvline(x = M, color = 'red', label = 'axvline - full height')
plt.show()

plt.plot(n, x, 'blue')
plt.plot(n, xp,'orange')
plt.axvline(x = M, color = 'red', label = 'axvline - full height')
plt.show()

plt.figure() #Espectro mascara percusiva
plt.specgram(xp, NFFT = N,Fs = fs)
#plt.yscale("log")
plt.ylim([0, 20000])
plt.grid()
plt.colorbar()
plt.show()

plt.figure() #Espectro mascara armonica
plt.specgram(xh, NFFT = N,Fs = fs)
#plt.yscale("log")
plt.ylim([0, 20000])
plt.grid()
plt.colorbar()
plt.show()



# %%
