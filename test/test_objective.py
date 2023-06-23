import numpy as np
from pytsmod import phase_vocoder

from utils.wav_utils import read_wav, save_wav
from test import plotting
from test import signals 
import pv
import pv_pl
import ola

def match_sizes(x1, x2):
    if len(x1) > len(x2):
        return x1[:len(x2)], x2
    else:
        return x1, x2[:len(x1)]

def basic_test(x_base, x_ideal, algo, plot, audio_save, return_audios, fs, N, Hs, alpha):
    """Test the algorithms on different test cases.

    Args:
        x_base (np.array): Base case signal of analysis.
        x_ideal (np.array): Ideal modification of the base signal. 
        algo (str): Name of the algorith to test. "PV", "PV_FL", "OLA", "TSM_HPS"
        plot (boolean): Option to display the result in a graph.
        audio_save (boolean): Option to save the audios in the audio directory.
        return_audios (boolean): Option to return the audio arrays.
        fs (int): Sample rate.
        N (int): Width of window.
        Hs (float): Synthesis hop size.
        alpha (float): Stretching factor.
    Returns:
        (float): Normalize value from 0 to 1 with the effectiveness of the algorithm.
        (float): Max amplitude diference between the signals.
    """
    if algo == "PV":
        #x_result = pv_2.TSM_PV(x_base, fs, N, alpha, Hs)
        x_result = pv.TSM_PV(x_base, fs, N, alpha, Hs)
        #x_result = pv_2.TSM_PV_copy(x_base, fs, N, alpha, Hs)
        
        #x_result, _ = pv_guille.TSM_PV(x_base, fs, N, alpha, Hs)
        x_ext = phase_vocoder(x_base, alpha, "hann", N, Hs, phase_lock=False)
    
    if algo == "PV_FL":
        x_result = pv_pl.TSM_PV_FL(x_base, fs, N, alpha, Hs)
    
    if algo == "OLA":
        x_result = ola.TSM_OLA(x_base, N, alpha, Hs)
    
    if algo == "TSM_HPS":
        x_result = None
    
    if plot:
        titles = ["Señal original", "Caso ideal", "Resultado"]
        plotting.compare_results(fs, titles, x_base, x_ideal, x_result)
    
    if audio_save:
        save_wav(x_base, fs, f"data\\test_base_{algo}.wav")
        save_wav(x_ideal, fs, f"data\\test_ideal_{algo}.wav")
        save_wav(x_result, fs, f"data\\test_result_{algo}.wav")
    
    if return_audios:
        return x_base, x_ideal, x_result

    x_ideal, x_result = match_sizes(x_ideal, x_result)
    gain_discrepancy = np.abs(np.max(x_ideal) - np.max(x_result))
        
    ##Mean absolute error
    similarity = np.sum(np.abs(x_ideal - x_result))/len(x_ideal)
    
    ##Euclidean distance
    #similarity = 1/len(x_ideal) * np.sum((x_ideal - x_result)**2)

    ##Square Euclidean distance
    #similarity = np.std(x_ideal - x_result) + (np.mean(x_ideal) - np.mean(x_result))**2

    ##Varational diference
    #similarity = 1/2 * (np.std(x_ideal - x_result)/(np.std(x_ideal) + np.std(x_result)))

    return similarity, gain_discrepancy
    
def test_ideal_signal(algo, f0, plot, audio_save, return_audios, fs, N, Hs, alpha):
    time = 1
    x_base, x_ideal = signals.simple_sine(f0, fs, alpha, time=time)
    sim_result = basic_test(x_base, x_ideal, algo, plot, audio_save, return_audios, fs, N, Hs, alpha)
    return sim_result

def test_freq_change_signal(algo, plot, audio_save, return_audios, fs, N, Hs, alpha):
    f1 = 200
    f2 = 450 
    t_fluct = 0.05
    time = 0.2
    x_base, x_ideal = signals.fast_changes(f1, f2, fs, t_fluct, alpha, time=time)
    sim_result = basic_test(x_base, x_ideal, algo, plot, audio_save, return_audios, fs, N, Hs, alpha)
    return sim_result

#Test PV
cfg1 = {"N": 4096, "Hs": 4096//4, "alpha": 1.2, "fs": 22050}

#OLA
cfg2 = {"N": 1024, "Hs": 1024//2, "alpha": 0.7, "fs": 22050}

#rta = test_ideal_signal("PV_FL", 500, plot=False, audio_save=False, return_audios=False, **cfg1)


#print(rta)