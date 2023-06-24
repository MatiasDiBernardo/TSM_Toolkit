import numpy as np

from main import apply_algo 
from utils.wav_utils import save_wav
from test import plotting
from test import signals 
from hps import hps

def match_sizes(x1, x2):
    if len(x1) > len(x2):
        return x1[:len(x2)], x2
    else:
        return x1, x2[:len(x1)]

def basic_test(x_base, x_ideal, fs, algo, config, plot, audio_save, return_audios):
    """Test the algorithms on different test cases.

    Args:
        x_base (np.array): Base case signal of analysis.
        x_ideal (np.array): Ideal modification of the base signal.
        fs (int): Sample rate.
        algo (str): Name of the algorith to test. "PV", "PV_FL", "OLA", "TSM_HPS"
        config (dicc): Config dictionary matching algo type.
        plot (boolean): Option to display the result in a graph.
        audio_save (boolean): Option to save the audios in the audio directory.
        return_audios (boolean): Option to return the audio arrays.
    Returns:
        (float): Normalize value from 0 to 1 with the effectiveness of the algorithm.
        (float): Max amplitude diference between the signals.
    """

    x_result = apply_algo(x_base, algo, config)
    
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
    
def test_ideal_signal(algo, f0, fs, config, plot, audio_save, return_audios):
    time = 1
    if algo == "HPS":
        cfg = config[1]
        alpha = cfg["alpha"]
    else:
        alpha = config["alpha"]
    x_base, x_ideal = signals.simple_sine(f0, fs, alpha, time=time)
    sim_result = basic_test(x_base, x_ideal, fs, algo, config, plot, audio_save, return_audios)
    
    return sim_result

def test_imp_signal(algo,fs, config, plot, audio_save, return_audios):
    time = 1
    if algo == "HPS":
        cfg = config[1]
        alpha = cfg["alpha"]
    else:
        alpha = config["alpha"]
        
    x_base, x_ideal = signals.simple_impulse(fs, alpha, time=time)
    sim_result = basic_test(x_base, x_ideal, fs, algo, config, plot, audio_save, return_audios)
    
    return sim_result 
    
def test_freq_change_signal(algo, fs, config, plot, audio_save, return_audios):
    f1 = 200
    f2 = 450 
    t_fluct = 0.05
    time = 0.2
    if algo == "HPS":
        cfg = config[1]
        alpha = cfg["alpha"]
    else:
        alpha = config["alpha"]
    x_base, x_ideal = signals.fast_changes(f1, f2, fs, t_fluct, alpha, time=time)
    sim_result = basic_test(x_base, x_ideal, fs, algo, config, plot, audio_save, return_audios)
    
    return sim_result

def dif_and_gain_values(algorithm, fs, configs):
    v_dif = []
    g_dif = []

    res = 50
    freqs = np.linspace(20,10000, res)  #Frecuencias a analizar

    for cfg in configs:

        sim_value = []
        gain_dif_value = []

        for f in freqs:
            sim, gain = test_ideal_signal(algo=algorithm, f0=f, fs=fs, config=cfg, plot=False, audio_save=False, return_audios=False)
            sim_value.append(sim)
            gain_dif_value.append(gain)
        
        sim_avg = np.sum(sim_value)/res
        gain_avg = np.sum(gain_dif_value)/res

        v_dif.append(sim_avg)
        g_dif.append(gain_avg)
    
    return v_dif, g_dif

def obj_test():
    """Compares diferent configurations on the ideal signal test
    calculating the avaregue of results between different frequencies.
    """

    fs = 22050
    algo = "PV"
    alpha = 0.5
    Hs_factor = 4

    cfg1 = {"N": 512, "Hs": 512//Hs_factor, "alpha": alpha, "fs": fs}
    cfg2 = {"N": 1024, "Hs": 1024//Hs_factor, "alpha": alpha, "fs": fs}
    cfg3 = {"N": 2048, "Hs": 2048//Hs_factor, "alpha": alpha, "fs": fs}
    cfg4 = {"N": 4096, "Hs": 4096//Hs_factor, "alpha": alpha, "fs": fs}
    cfg5 = {"N": 8192, "Hs": 8192//Hs_factor, "alpha": alpha, "fs": fs}
    cfg6 = {"N": 16384, "Hs": 16384//Hs_factor, "alpha": alpha, "fs": fs}

    configs = [cfg1, cfg2, cfg3, cfg4, cfg5, cfg6]

    dif, gain = dif_and_gain_values(algo, fs, configs)
    print("Dif ideal: ", dif)
    print("Dif gain: ", gain)

def hps_similarity_test(xp,xh,plot, N, M, fs):
    """Measure the similarity between pure aromonic and percuse signal imput and the hps output.

    Args:
        xp(np.array): Percusive signal.
        x_ideal (np.array): Harmonic signal. 
        plot (boolean): Option to display the result in a graph.
        audio_save (boolean): Option to save the audios in the audio directory.
        N (int): STFT lenght.
        N (int): Median filter lenght (must be od)
        fs (int): Sample rate.
    Returns:
        Percusive_Similarity (float): Normalize value from 0 to 1 with the effectiveness of the algorithm.  
        Harmonic_Similarity (float): Normalize value from 0 to 1 with the effectiveness of the algorithm.       
    """
    x = xp + xh
    yp, yh = hps(x,N,M)
    h_similarity = np.sum(np.abs(xp - yp))/len(yp)
    p_similarity = np.sum(np.abs(xh - yh))/len(yh)
    harmonic_max_sim = np.abs(np.max(xh) - np.max(yh))/np.max(xh)
    return h_similarity, p_similarity, harmonic_max_sim
