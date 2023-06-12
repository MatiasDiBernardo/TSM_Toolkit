import matplotlib.pyplot as plt
import numpy as np

def compare_3_results(x1, x2, x3, fs):
    """Plot three signal in a graph.

    Args:
        x1 (np.array): Original signal.
        x2 (np.array): Ideal signal
        x3 (np.array): Modified signal.
        fs (int): Sample rate.
    """
    
    fig, ax = plt.subplots(3, 1)
    t1 = np.linspace(0, len(x1)/fs, len(x1))
    ax[0].plot(t1, x1)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time")

    t2 = np.linspace(0, len(x2)/fs, len(x2))
    ax[1].plot(t2, x2)
    ax[1].set_ylabel("Amplitude")
    ax[1].set_xlabel("Time")

    t3 = np.linspace(0, len(x3)/fs, len(x3))
    ax[2].plot(t3, x3)
    ax[2].set_ylabel("Amplitude")
    ax[2].set_xlabel("Time")

    plt.show()
    
def basic_plot_comparison(x1,x2,fs):
    """Plot two signals. Compare the signal before and after applying the TSM algorithm.

    Args:
        x1 (np.array): Original signal.
        x2 (np.array): Modified signal.
        fs (int): Sample rate.
    """
    fig, ax = plt.subplots(2, 1)

    t1 = np.linspace(0, len(x1)/fs, len(x1))
    ax[0].set_title("Señal original vs Señal con TSM")
    ax[0].plot(t1, x1)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time")
    if len(x1)/fs > len(x2)/fs:
        ax[0].setxlim([0,len(x1)/fs])
    else:
        ax[0].setxlim([0,len(x2)/fs])

    t2 = np.linspace(0, len(x2)/fs, len(x2))
    ax[1].plot(t2, x2)
    ax[1].set_ylabel("Amplitude")
    ax[1].set_xlabel("Time")
    if len(x1)/fs > len(x2)/fs:
        ax[1].setxlim([0,len(x1)/fs])
    else:
        ax[1].setxlim([0,len(x2)/fs])

    plt.show()

    plt.show()
    



