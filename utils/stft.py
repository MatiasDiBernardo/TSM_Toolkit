import librosa
import numpy as np
import matplotlib.pyplot as plt

def compute_stft(x, n_fft, hop_size, win_length, win_type):
    """Computs stft of a given signal.

    Args:
        x (np.array): Audio signal
        n_fft (int): Lenght of fft for each bean. Padded if needed.
        hop_size (int): Value of overlapping between frames.
        win_length (int): Length of internal windwing.
        win_type (str): Type of window, ex: Hann, Hamming, Blackman.

    Return:
        mag(np.array): Ampitude of the stft.
        phase(np.arrat): phase of the stft
    """

    assert len(x) > win_length, "Segment to short for stft"
    assert hop_size < win_length//2, "Hop size needs to be small"

    stft_matrix = librosa.stft(x, n_fft=n_fft, hop_length=hop_size, win_length=win_length, window=win_type)

    mag = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    return mag, phase

def compute_istft(mag, phase, n_fft, hop_size, win_length, win_type):
    """Computes the inverse stft to obtain original time series. All
    parameters should be the same as the original stft.

    Args:
        mag(np.array): Ampitude of the stft.
        phase(np.arrat): phase of the stft
        n_fft (int): Lenght of fft for each bean. Padded if needed.
        hop_size (int): Value of overlapping between frames.
        win_length (int): Length of internal windwing.
        win_type (str): Type of window, ex: Hann, Hamming, Blackman.

    Return:
        y(np.array): Original time series.
    """

    stft_matrix = mag * np.exp(1j * phase)
    y = librosa.istft(stft_matrix=stft_matrix, hop_length=hop_size, n_fft=n_fft, win_length=win_length, window=win_type)

    return y

def plot_stft(mag):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(mag, ref=np.max),
    y_axis='log', x_axis='time', ax=ax)

    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
