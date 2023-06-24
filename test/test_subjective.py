from utils.wav_utils import read_wav,save_wav
from main import apply_algo
from test import plotting

def compare_algorithms(path_audio, algo_comp, plot, save_audios, return_audios):
    """Compare a real audio with different algorithms or different parameters.
    Algo types: "OLA" : Overlapp-add
                "PV" : Phase Vocoder
                "PV_FL" : Phase Vocoder with Phase Locking
                "HPS_SOLO" : Harmonic-percussive separation
                "HPS" : TSM with harmonic-percussive separation
                "PV_REG" : Pytsmod Implementation Phase Vocoder
                "SIN_M" : TSM base on Sine Model.

    Args:
        path_audio (str): Path to audio file.
        algo_comp (list): List with type of algo, config and title. Ex ["PV", cgf1, "PV Cfg1"]
        plot (boolean): Plot or not.
        save_audios (bolean): Save audios or not.
        return_audios (bolean): Return audios or not.
    """

    fs = 22050
    test, _ = read_wav(path_audio, fs, mono=True)

    titles = []
    signals = []

    for data in algo_comp:
        y = apply_algo(test, data[0], data[1])
        signals.append(y)
        titles.append(data[2])
        
    if plot:
        plotting.compare_results(fs, titles, *signals)
    
    if save_audios:
        name = path_audio.split(".")[0]
        for i in range(len(signals)):
            save_wav(signals[i], fs, f"{name}_{titles[i]}.wav" )
    
    if return_audios:
        return signals
 