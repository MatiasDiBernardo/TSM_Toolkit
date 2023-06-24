import matplotlib.pyplot as plt
import numpy as np

def basic_plot(x1, ax, fs, index, titles):
    t1 = np.linspace(0, len(x1)/fs, len(x1))
    ax[index].set_title(titles[index])
    ax[index].plot(t1, x1)
    ax[index].set_ylabel("Amplitude")
    #ax[index].set_xlabel("Time")

def compare_results(fs, titles, *x):
    """Compare signals graphicaly in a plot.

    Args:
        fs (int): Sample rate.
        titles(str): List of names. Must be same size of amount of signals.
        x* (np.array): Signal to plot.
    """
    fig, ax = plt.subplots(len(x), 1)

    for i in range(len(x)):
        basic_plot(x[i], ax, fs, i, titles)
    
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    fig.subplots_adjust(hspace=1)

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


def basic_3plot(x1,x2,x3,fs):
    """Plot three signals. 
    Compares the original signal with the refrence (pytsmod pack) 
    with the local tsm algorithm.
    it keeps the horizontal axis according to the longer signal, 
    for an easier visual comparison.
    Args:
        x1 (np.array): Original signal.
        x2 (np.array): Modified signal with pytsmod.
        x3 (np.array): Modified signal with our code.
        fs (int): Sample rate.
    """

    #calculate the x axis time array
    lengths = [len(x1), len(x2), len(x3)]
    max_length = max(lengths)

    fig, ax = plt.subplots(3, 1)

    t1 = np.linspace(0, len(x1)/fs, len(x1))
    ax[0].set_title("Señal original")
    ax[0].plot(t1, x1)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time") 
    ax[0].set_xlim([0,max_length/fs])

    t2 = np.linspace(0, len(x2)/fs, len(x2))
    ax[1].set_title("Señal modificada con pytsmod (ref package)")
    ax[1].plot(t2, x2)
    ax[1].set_ylabel("Amplitude")
    ax[1].set_xlabel("Time")
    ax[1].set_xlim([0,max_length/fs])

    t3 = np.linspace(0, len(x3)/fs, len(x3))
    ax[2].set_title("Señal modificada con nuestro codigo")
    ax[2].plot(t3, x3)
    ax[2].set_ylabel("Amplitude")
    ax[2].set_xlabel("Time")
    ax[2].set_xlim([0,max_length/fs])

