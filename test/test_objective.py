import numpy as np

from utils.wav_utils import read_wav, save_wav
import plotting
import signals
import pv
import pv_2
import pv_3
import pv_phase_locking
import ola

def basic_test(x_base, x_ideal, algo, plot, audio_save, fs, N, Hs, alpha):
    """Test the algorithms on different test cases.

    Args:
        x_base (np.array): Base case signal of analysis.
        x_ideal (np.array): Ideal modification of the base signal. 
        algo (str): Name of the algorith to test. "PV", "PV_FL", "OLA", "TSM_HPS"
        plot (boolean): Option to display the result in a graph.
        audio_save (boolean): Option to save the audios in the audio directory.
        fs (int): Sample rate.
        N (int): Width of window.
        Hs (float): Synthesis hop size.
        alpha (float): Stretching factor.
    Returns:
        (float): Normalize value from 0 to 1 with the effectiveness of the algorithm.
    """
    if algo == "PV":
        x_result = pv.TSM_PV(x_base, fs, N, alpha, Hs)
        #x_result = pv_2.TSM_PV(x_base, fs, N, alpha, Hs)
        #x_result = pv_3.TSM_PV(x_base, fs, N, alpha, Hs)
    
    if algo == "PV_FL":
        x_result = pv_phase_locking.TSM_PV_phase_locking(x_base, fs, N, alpha, Hs)
    
    if algo == "OLA":
        x_result = ola.TSM_OLA(x_base, N, alpha, Hs)
    
    if algo == "TSM_HPS":
        x_result = None
    
    if plot:
        plotting.compare_3_results(x_base, x_ideal, x_result, fs)
    
    if audio_save:
        save_wav(x_base, fs, f"data\\test_base_{algo}.wav")
        save_wav(x_ideal, fs, f"data\\test_ideal_{algo}.wav")
        save_wav(x_result, fs, f"data\\test_result_{algo}.wav")
        
    if alpha < 1:
        x_ideal = np.concatenate([x_ideal, np.zeros(N)])
    
    similarity = 1/N * np.sqrt(x_ideal**2 + x_result**2)
    
    return np.sum(similarity)

def test_ideal_signal(algo, plot, audio_save, fs, N, Hs, alpha):
    f0 = 500
    time = 1
    x_base, x_ideal = signals.simple_sine(f0, fs, alpha, time=time)
    sim_result = basic_test(x_base, x_ideal, algo, plot, audio_save, fs, N, Hs, alpha)
    return sim_result

def test_freq_change_signal(algo, plot, audio_save, fs, N, Hs, alpha):
    f1 = 200
    f2 = 450 
    t_fluct = 0.05
    time = 0.2
    x_base, x_ideal = signals.fast_changes(f1, f2, fs, t_fluct, alpha, time=time)
    sim_result = basic_test(x_base, x_ideal, algo, plot, audio_save, fs, N, Hs, alpha)
    return sim_result

#Test PV
cfg1 = {"N": 2048, "Hs": 2048//4, "alpha": 1.5, "fs": 22050}

rta = test_freq_change_signal("PV", plot=True, audio_save=False, **cfg1) 
print("Test: ", rta)